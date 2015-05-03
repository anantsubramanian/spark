/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.recommendation

import java.{util => ju}

import scala.collection.mutable
import scala.reflect.ClassTag
import scala.util.Sorting
import scala.util.hashing.byteswap64

import com.github.fommil.netlib.BLAS.{getInstance => blas}
import com.github.fommil.netlib.LAPACK.{getInstance => lapack}
import org.jblas.DoubleMatrix
import org.netlib.util.intW

import org.apache.spark.{Logging, Partitioner}
import org.apache.spark.annotation.DeveloperApi
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.param._
import org.apache.spark.mllib.optimization.NNLS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, FloatType, IntegerType, StructField, StructType}
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.Utils
import org.apache.spark.util.collection.{OpenHashMap, OpenHashSet, SortDataFormat, Sorter}
import org.apache.spark.util.random.XORShiftRandom

/**
 * Common params for ALS.
 */
private[recommendation] trait ALSParams extends Params with HasMaxIter with HasRegParam
  with HasPredictionCol {

  /**
   * Param for rank of the matrix factorization.
   * @group param
   */
  val rank = new IntParam(this, "rank", "rank of the factorization", Some(10))

  /** @group getParam */
  def getRank: Int = get(rank)

  /**
   * Param for number of user blocks.
   * @group param
   */
  val numUserBlocks = new IntParam(this, "numUserBlocks", "number of user blocks", Some(10))

  /** @group getParam */
  def getNumUserBlocks: Int = get(numUserBlocks)

  /**
   * Param for number of item blocks.
   * @group param
   */
  val numItemBlocks =
    new IntParam(this, "numItemBlocks", "number of item blocks", Some(10))

  /** @group getParam */
  def getNumItemBlocks: Int = get(numItemBlocks)

  /**
   * Param to decide whether to use implicit preference.
   * @group param
   */
  val implicitPrefs =
    new BooleanParam(this, "implicitPrefs", "whether to use implicit preference", Some(false))

  /** @group getParam */
  def getImplicitPrefs: Boolean = get(implicitPrefs)

  /**
   * Param for the alpha parameter in the implicit preference formulation.
   * @group param
   */
  val alpha = new DoubleParam(this, "alpha", "alpha for implicit preference", Some(1.0))

  /** @group getParam */
  def getAlpha: Double = get(alpha)

  /**
   * Param for the column name for user ids.
   * @group param
   */
  val userCol = new Param[String](this, "userCol", "column name for user ids", Some("user"))

  /** @group getParam */
  def getUserCol: String = get(userCol)

  /**
   * Param for the column name for item ids.
   * @group param
   */
  val itemCol =
    new Param[String](this, "itemCol", "column name for item ids", Some("item"))

  /** @group getParam */
  def getItemCol: String = get(itemCol)

  /**
   * Param for the column name for ratings.
   * @group param
   */
  val ratingCol = new Param[String](this, "ratingCol", "column name for ratings", Some("rating"))

  /** @group getParam */
  def getRatingCol: String = get(ratingCol)

  /**
   * Param for whether to apply nonnegativity constraints.
   * @group param
   */
  val nonnegative = new BooleanParam(
    this, "nonnegative", "whether to use nonnegative constraint for least squares", Some(false))

  /** @group getParam */
  val getNonnegative: Boolean = get(nonnegative)

  /**
   * Validates and transforms the input schema.
   * @param schema input schema
   * @param paramMap extra params
   * @return output schema
   */
  protected def validateAndTransformSchema(schema: StructType, paramMap: ParamMap): StructType = {
    val map = this.paramMap ++ paramMap
    assert(schema(map(userCol)).dataType == IntegerType)
    assert(schema(map(itemCol)).dataType== IntegerType)
    val ratingType = schema(map(ratingCol)).dataType
    assert(ratingType == FloatType || ratingType == DoubleType)
    val predictionColName = map(predictionCol)
    assert(!schema.fieldNames.contains(predictionColName),
      s"Prediction column $predictionColName already exists.")
    val newFields = schema.fields :+ StructField(map(predictionCol), FloatType, nullable = false)
    StructType(newFields)
  }
}

/**
 * Model fitted by ALS.
 */
class ALSModel private[ml] (
    override val parent: ALS,
    override val fittingParamMap: ParamMap,
    k: Int,
    userFactors: RDD[(Int, Array[Float])],
    itemFactors: RDD[(Int, Array[Float])])
  extends Model[ALSModel] with ALSParams {

  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  override def transform(dataset: DataFrame, paramMap: ParamMap): DataFrame = {
    import dataset.sqlContext.implicits._
    val map = this.paramMap ++ paramMap
    val users = userFactors.toDF("id", "features")
    val items = itemFactors.toDF("id", "features")

    // Register a UDF for DataFrame, and then
    // create a new column named map(predictionCol) by running the predict UDF.
    val predict = udf { (userFeatures: Seq[Float], itemFeatures: Seq[Float]) =>
      if (userFeatures != null && itemFeatures != null) {
        blas.sdot(k, userFeatures.toArray, 1, itemFeatures.toArray, 1)
      } else {
        Float.NaN
      }
    }
    dataset
      .join(users, dataset(map(userCol)) === users("id"), "left")
      .join(items, dataset(map(itemCol)) === items("id"), "left")
      .select(dataset("*"), predict(users("features"), items("features")).as(map(predictionCol)))
  }

  override def transformSchema(schema: StructType, paramMap: ParamMap): StructType = {
    validateAndTransformSchema(schema, paramMap)
  }
}


/**
 * Alternating Least Squares (ALS) matrix factorization.
 *
 * ALS attempts to estimate the ratings matrix `R` as the product of two lower-rank matrices,
 * `X` and `Y`, i.e. `X * Yt = R`. Typically these approximations are called 'factor' matrices.
 * The general approach is iterative. During each iteration, one of the factor matrices is held
 * constant, while the other is solved for using least squares. The newly-solved factor matrix is
 * then held constant while solving for the other factor matrix.
 *
 * This is a blocked implementation of the ALS factorization algorithm that groups the two sets
 * of factors (referred to as "users" and "products") into blocks and reduces communication by only
 * sending one copy of each user vector to each product block on each iteration, and only for the
 * product blocks that need that user's feature vector. This is achieved by pre-computing some
 * information about the ratings matrix to determine the "out-links" of each user (which blocks of
 * products it will contribute to) and "in-link" information for each product (which of the feature
 * vectors it receives from each user block it will depend on). This allows us to send only an
 * array of feature vectors between each user block and product block, and have the product block
 * find the users' ratings and update the products based on these messages.
 *
 * For implicit preference data, the algorithm used is based on
 * "Collaborative Filtering for Implicit Feedback Datasets", available at
 * [[http://dx.doi.org/10.1109/ICDM.2008.22]], adapted for the blocked approach used here.
 *
 * Essentially instead of finding the low-rank approximations to the rating matrix `R`,
 * this finds the approximations for a preference matrix `P` where the elements of `P` are 1 if
 * r > 0 and 0 if r <= 0. The ratings then act as 'confidence' values related to strength of
 * indicated user
 * preferences rather than explicit ratings given to items.
 */
class ALS extends Estimator[ALSModel] with ALSParams {

  import org.apache.spark.ml.recommendation.ALS.Rating

  /** @group setParam */
  def setRank(value: Int): this.type = set(rank, value)

  /** @group setParam */
  def setNumUserBlocks(value: Int): this.type = set(numUserBlocks, value)

  /** @group setParam */
  def setNumItemBlocks(value: Int): this.type = set(numItemBlocks, value)

  /** @group setParam */
  def setImplicitPrefs(value: Boolean): this.type = set(implicitPrefs, value)

  /** @group setParam */
  def setAlpha(value: Double): this.type = set(alpha, value)

  /** @group setParam */
  def setUserCol(value: String): this.type = set(userCol, value)

  /** @group setParam */
  def setItemCol(value: String): this.type = set(itemCol, value)

  /** @group setParam */
  def setRatingCol(value: String): this.type = set(ratingCol, value)

  /** @group setParam */
  def setPredictionCol(value: String): this.type = set(predictionCol, value)

  /** @group setParam */
  def setMaxIter(value: Int): this.type = set(maxIter, value)

  /** @group setParam */
  def setRegParam(value: Double): this.type = set(regParam, value)

  /** @group setParam */
  def setNonnegative(value: Boolean): this.type = set(nonnegative, value)

  /**
   * Sets both numUserBlocks and numItemBlocks to the specific value.
   * @group setParam
   */
  def setNumBlocks(value: Int): this.type = {
    setNumUserBlocks(value)
    setNumItemBlocks(value)
    this
  }

  setMaxIter(20)
  setRegParam(1.0)

  override def fit(dataset: DataFrame, paramMap: ParamMap): ALSModel = {
    val map = this.paramMap ++ paramMap
    val ratings = dataset
      .select(col(map(userCol)), col(map(itemCol)), col(map(ratingCol)).cast(FloatType))
      .map { row =>
        Rating(row.getInt(0), row.getInt(1), row.getFloat(2))
      }
    val (userFactors, itemFactors) = ALS.train(ratings, rank = map(rank),
      numUserBlocks = map(numUserBlocks), numItemBlocks = map(numItemBlocks),
      maxIter = map(maxIter), regParam = map(regParam), implicitPrefs = map(implicitPrefs),
      alpha = map(alpha), nonnegative = map(nonnegative))
    val model = new ALSModel(this, map, map(rank), userFactors, itemFactors)
    Params.inheritValues(map, this, model)
    model
  }

  /*def fitPNCG(dataset: DataFrame, paramMap: ParamMap): ALSModel = {*/
  def fitPNCG(dataset: DataFrame): ALSModel = {
    /*val map = this.paramMap ++ paramMap*/
    val map = this.paramMap 
    val ratings = dataset
      .select(col(map(userCol)), col(map(itemCol)), col(map(ratingCol)).cast(FloatType))
      .map { row =>
        Rating(row.getInt(0), row.getInt(1), row.getFloat(2))
      }
    val (userFactors, itemFactors) = ALS.trainPNCG(ratings, rank = map(rank),
      numUserBlocks = map(numUserBlocks), numItemBlocks = map(numItemBlocks),
      maxIter = map(maxIter), regParam = map(regParam), implicitPrefs = map(implicitPrefs),
      alpha = map(alpha), nonnegative = map(nonnegative))
    val model = new ALSModel(this, map, map(rank), userFactors, itemFactors)
    Params.inheritValues(map, this, model)
    model
  }

  override def transformSchema(schema: StructType, paramMap: ParamMap): StructType = {
    validateAndTransformSchema(schema, paramMap)
  }
}

/**
 * :: DeveloperApi ::
 * An implementation of ALS that supports generic ID types, specialized for Int and Long. This is
 * exposed as a developer API for users who do need other ID types. But it is not recommended
 * because it increases the shuffle size and memory requirement during training. For simplicity,
 * users and items must have the same type. The number of distinct users/items should be smaller
 * than 2 billion.
 */
@DeveloperApi
object ALS extends Logging {

  /** Rating class for better code readability. */
  case class Rating[@specialized(Int, Long) ID](user: ID, item: ID, rating: Float)

  /** Trait for least squares solvers applied to the normal equation. */
  private[recommendation] trait LeastSquaresNESolver extends Serializable {
    /** Solves a least squares problem (possibly with other constraints). */
    def solve(ne: NormalEquation, lambda: Double): Array[Float]
  }
  private def logStdout(msg: String): Unit = {
		val time: Long = System.currentTimeMillis;
		logInfo(msg);
		println(time + ": " + msg);
	}

  /** Cholesky solver for least square problems. */
  private[recommendation] class CholeskySolver extends LeastSquaresNESolver {

    private val upper = "U"

    /**
     * Solves a least squares problem with L2 regularization:
     *
     *   min norm(A x - b)^2^ + lambda * n * norm(x)^2^
     *
     * @param ne a [[NormalEquation]] instance that contains AtA, Atb, and n (number of instances)
     * @param lambda regularization constant, which will be scaled by n
     * @return the solution x
     */
    override def solve(ne: NormalEquation, lambda: Double): Array[Float] = {
      val k = ne.k
      // Add scaled lambda to the diagonals of AtA.
      val scaledlambda = lambda * ne.n
      var i = 0
      var j = 2
      while (i < ne.triK) {
        ne.ata(i) += scaledlambda
        i += j
        j += 1
      }
      val info = new intW(0)
      lapack.dppsv(upper, k, 1, ne.ata, ne.atb, k, info)
      val code = info.`val`
      assert(code == 0, s"lapack.dppsv returned $code.")
      val x = new Array[Float](k)
      i = 0
      while (i < k) {
        x(i) = ne.atb(i).toFloat
        i += 1
      }
      ne.reset()
      x
    }
  }

  /** NNLS solver. */
  private[recommendation] class NNLSSolver extends LeastSquaresNESolver {
    private var rank: Int = -1
    private var workspace: NNLS.Workspace = _
    private var ata: DoubleMatrix = _
    private var initialized: Boolean = false

    private def initialize(rank: Int): Unit = {
      if (!initialized) {
        this.rank = rank
        workspace = NNLS.createWorkspace(rank)
        ata = new DoubleMatrix(rank, rank)
        initialized = true
      } else {
        require(this.rank == rank)
      }
    }

    /**
     * Solves a nonnegative least squares problem with L2 regularizatin:
     *
     *   min_x_  norm(A x - b)^2^ + lambda * n * norm(x)^2^
     *   subject to x >= 0
     */
    override def solve(ne: NormalEquation, lambda: Double): Array[Float] = {
      val rank = ne.k
      initialize(rank)
      fillAtA(ne.ata, lambda * ne.n)
      val x = NNLS.solve(ata, new DoubleMatrix(rank, 1, ne.atb: _*), workspace)
      ne.reset()
      x.map(x => x.toFloat)
    }

    /**
     * Given a triangular matrix in the order of fillXtX above, compute the full symmetric square
     * matrix that it represents, storing it into destMatrix.
     */
    private def fillAtA(triAtA: Array[Double], lambda: Double) {
      var i = 0
      var pos = 0
      var a = 0.0
      val data = ata.data
      while (i < rank) {
        var j = 0
        while (j <= i) {
          a = triAtA(pos)
          data(i * rank + j) = a
          data(j * rank + i) = a
          pos += 1
          j += 1
        }
        data(i * rank + i) += lambda
        i += 1
      }
    }
  }

  /** Representing a normal equation (ALS' subproblem). */
  private[recommendation] class NormalEquation(val k: Int) extends Serializable {

    /** Number of entries in the upper triangular part of a k-by-k matrix. */
    val triK = k * (k + 1) / 2
    /** A^T^ * A */
    val ata = new Array[Double](triK)
    /** A^T^ * b */
    val atb = new Array[Double](k)
    /** Number of observations. */
    var n = 0

    private val da = new Array[Double](k)
    private val upper = "U"

    private def copyToDouble(a: Array[Float]): Unit = {
      var i = 0
      while (i < k) {
        da(i) = a(i)
        i += 1
      }
    }

    /** Adds an observation. */
    def add(a: Array[Float], b: Float): this.type = {
      require(a.length == k)
      copyToDouble(a)
      blas.dspr(upper, k, 1.0, da, 1, ata)
      blas.daxpy(k, b.toDouble, da, 1, atb, 1)
      n += 1
      this
    }

    /**
     * Adds an observation with implicit feedback. Note that this does not increment the counter.
     */
    def addImplicit(a: Array[Float], b: Float, alpha: Double): this.type = {
      require(a.length == k)
      // Extension to the original paper to handle b < 0. confidence is a function of |b| instead
      // so that it is never negative.
      val confidence = 1.0 + alpha * math.abs(b)
      copyToDouble(a)
      blas.dspr(upper, k, confidence - 1.0, da, 1, ata)
      // For b <= 0, the corresponding preference is 0. So the term below is only added for b > 0.
      if (b > 0) {
        blas.daxpy(k, confidence, da, 1, atb, 1)
      }
      this
    }

    /** Merges another normal equation object. */
    def merge(other: NormalEquation): this.type = {
      require(other.k == k)
      blas.daxpy(ata.length, 1.0, other.ata, 1, ata, 1)
      blas.daxpy(atb.length, 1.0, other.atb, 1, atb, 1)
      n += other.n
      this
    }

    /** Resets everything to zero, which should be called after each solve. */
    def reset(): Unit = {
      ju.Arrays.fill(ata, 0.0)
      ju.Arrays.fill(atb, 0.0)
      n = 0
    }
  }

  /**
   * Backtracking linesearch for type [X] and f(X) = Double
   *
   */
  def linesearch[X](
      func: (X) => Float, 
      axpy: (Float) => X,
      x0: X, 
      direc: X,
      reduceFrac: Float,
      initStep: Float,
      dirProdGrad: Float,
      maxIters: Int = 10
      ): Float = 
  {
    val f0: Float = func(x0)

    var step: Float = initStep
    var x: X = axpy(step)
    var f: Float = func(x)
    var k: Int = 1;

    while ( (f - f0 > step*dirProdGrad) && (k <= maxIters) )
    {
      // x = a*direc + x0
      step = reduceFrac * step
      x = axpy(step)
      f = func(x)
      k += 1
    }
    step
  }
    

  /**
   * Implementation of the Nonlinear Preconditioned Conjugate Gradient (PNCG) 
   * ALS is used as a preconditioner to the standard nonlinear CG.
   *
   */

  def trainPNCG[ID: ClassTag]( // scalastyle:ignore
      ratings: RDD[Rating[ID]],
      rank: Int = 10,
      numUserBlocks: Int = 10,
      numItemBlocks: Int = 10,
      maxIter: Int = 10,
      regParam: Double = 1.0,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      nonnegative: Boolean = false,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      seed: Long = 0L)(
      implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = 
  {
    require(intermediateRDDStorageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    val sc = ratings.sparkContext

    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)

    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)

    val solver = {
      if (nonnegative) 
        new NNLSSolver 
      else 
        new CholeskySolver
    }

    val blockRatings: RDD[((Int, Int), RatingBlock[ID])] = 
      partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)

    val (userInBlocks, userOutBlocks, userCounts) = makeBlocks("user", 
      blockRatings, 
      userPart, 
      itemPart, 
      intermediateRDDStorageLevel)

    // materialize blockRatings and user blocks
    userOutBlocks.count()
    userCounts.count()

    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks, itemCounts) = makeBlocks("item", 
      swappedBlockRatings, 
      itemPart, 
      userPart, 
      intermediateRDDStorageLevel)

    // materialize item blocks
    itemOutBlocks.count()
    itemCounts.count()


    type FactorRDD = RDD[(Int,FactorBlock)]

    def preconditionItems(u: FactorRDD): FactorRDD = 
    {
      computeFactors(u, 
        userOutBlocks, 
        itemInBlocks, 
        rank, 
        regParam,
        userLocalIndexEncoder, 
        solver = solver)
    }

    def preconditionUsers(m: FactorRDD): FactorRDD = 
    {
      computeFactors(m, 
        itemOutBlocks, 
        userInBlocks, 
        rank, 
        regParam,
        itemLocalIndexEncoder, 
        solver = solver)
    }

    /** 
     * Compute blas.SCALE; x = a*x, and return a new factor block 
     * @param x a vector represented as a FactorBlock
     * @param a scalar
     */
    def blockSCAL(x: FactorBlock, a: Float): FactorBlock = 
    {
      val numVectors = x.length
      /*val result = Array.ofDim[Array[Float]](numVectors)*/
      /*Array.copy(x,0,result,0,numVectors)*/
      val result: FactorBlock = x.clone()

      var k = 0;
      while (k < numVectors)
      {
        blas.sscal(rank,a,result(k),1)
        k += 1
      }
      result
    }

    /** 
     * compute a*x + b*y, and return a new factor block 
     * @param x a vector represented as a FactorBlock
     * @param y a vector represented as a FactorBlock
     * @param a a scalar
     */
    def blockAXPBY(a: Float, x: FactorBlock, b: Float, y: FactorBlock): FactorBlock =
    {
      val numVectors = x.length
      /*val result: FactorBlock = Array.ofDim[Array[Float]](numVectors)*/
      /*Array.copy(y,0,result,0,numVectors)*/
      val result: FactorBlock = y.map(_.clone())

      var k = 0;
      while (k < numVectors)
      {
        //first scale b*y
        blas.sscal(rank,b,result(k),1)
        //y := a*x + (b*y)
        blas.saxpy(rank,a,x(k),1,result(k),1)
        k += 1
      }
      result
    }

    /** 
     * compute a*x + y, and return a new factor block 
     * @param a a scalar
     * @param x a vector represented as a FactorBlock
     * @param y a vector represented as a FactorBlock
     */
    def blockAXPY(a: Float, x: FactorBlock, y: FactorBlock): FactorBlock =
    {
      val numVectors = x.length
      /*val result: FactorBlock = Array.ofDim[Array[Float]](numVectors)*/
      /*Array.copy(y,0,result,0,numVectors)*/
      val result: FactorBlock = y.map(_.clone())

      var k = 0
      while (k < numVectors)
      {
        blas.saxpy(rank,a,x(k),1,result(k),1)
        k += 1
      }
      result
    }

    /** 
     * compute x dot y, and return a scalar
     * @param x a vector represented as a FactorBlock
     * @param y a vector represented as a FactorBlock
     */
    def blockDOT(x: FactorBlock, y: FactorBlock): Float = 
    {
      val numVectors = x.length
      var result: Float = 0.0f
      var k = 0
      while (k < numVectors)
      {
        result += blas.sdot(rank,x(k),1,y(k),1)
        k += 1
      }
      result
    }

    /** 
     * compute x dot x, and return a scalar
     * @param x a vector represented as a FactorBlock
     */
    def blockNRMSQR(x: FactorBlock): Float = 
    {
      val numVectors = x.length
      var result: Float = 0.0f
      var norm: Float = 0.0f
      var k = 0
      while (k < numVectors)
      {
        /*norm = blas.snrm2(rank,x(k),1)*/
        /*result += norm * norm*/
        result += blas.sdot(rank,x(k),1,x(k),1)
        k += 1
      }
      result
    }

    /** 
     * compute dot product, and return a scalar
     * @param xs RDD of FactorBlocks
     * @param ys RDD of FactorBlocks
     */
    def rddDOT(xs: FactorRDD, ys: FactorRDD): Float = {
      xs.join(ys)
        .map{case (_,(x,y)) => blockDOT(x,y)}
        .reduce{_+_}
    }


    /** 
     * compute x dot x, and return a scalar
     * @param xs RDD of FactorBlocks
     */
    def rddNORMSQR(xs: FactorRDD): Float = {
      xs.map{case (_,x) => blockNRMSQR(x)}
        .reduce(_+_)
    }

    /** 
     * compute 2-norm of an RDD of FactorBlocks
     * @param xs RDD of FactorBlocks
     */
    def rddNORM2(xs: FactorRDD): Float = {
      math.sqrt(rddNORMSQR(xs).toDouble).toFloat
    }

    /** 
     * compute a*x + b*y, for a FactorRDD = RDD[(Int, FactorBlock)]
     * @param x an RDD of vectors represented as a FactorBlock
     * @param y an RDD of vectors represented as a FactorBlock
     * @param a a scalar
     * @param b a scalar
     */
    def rddAXPBY(a: Float, x: FactorRDD, b: Float, y: FactorRDD): FactorRDD = {
      x.join(y).mapValues{case(xblock,yblock) => blockAXPBY(a,xblock,b,yblock)}
    }

    /** 
     * compute a*x + y, for a FactorRDD = RDD[(Int, FactorBlock)]
     * @param x an RDD of vectors represented as a FactorBlock
     * @param y an RDD of vectors represented as a FactorBlock
     * @param a a scalar
     */
    def rddAXPY(a: Float, x: FactorRDD, y: FactorRDD): FactorRDD = 
    {
      x.join(y).mapValues{case (xblock,yblock) => 
        blockAXPY(a,xblock,yblock)
      }
    }

    type FacTup = (FactorRDD,FactorRDD) // (user,items)
    def costFunc(x: FacTup): Float =
    {
      val usr = x._1
      val itm = x._2
      val sumSquaredErr: Float = evalFrobeniusCost(
        itm, 
        usr, 
        itemOutBlocks, 
        userInBlocks, 
        rank, 
        regParam,
        itemLocalIndexEncoder
      )  
      val usrNorm: Float = evalTikhonovNorm(
        usr, 
        userCounts,
        rank,
        regParam
      ) 
      val itmNorm: Float = evalTikhonovNorm(
        itm, 
        itemCounts,
        rank,
        regParam
      )
      sumSquaredErr + usrNorm + itmNorm
    }

    def computeGradient(userFac: FactorRDD, itemFac: FactorRDD): (FactorRDD,FactorRDD) =
    {
      val userGrad: FactorRDD = evalGradient(
        itemFac, 
        userFac,
        itemOutBlocks, 
        userInBlocks, 
        rank, 
        regParam,
        itemLocalIndexEncoder
      ).cache()

      val itemGrad: FactorRDD = evalGradient(
        userFac,
        itemFac, 
        userOutBlocks, 
        itemInBlocks, 
        rank, 
        regParam,
        userLocalIndexEncoder
      ).cache()
      (userGrad,itemGrad)
    }

    def computeAlpha(
        userFac: FactorRDD,
        itemFac: FactorRDD,
        userDirec: FactorRDD,
        itemDirec: FactorRDD,
        initStep: Float,
        reduceFrac: Float,
        gradFrac: Float,
        maxIters: Int): Float = 
    {
      /*def axpy(a: Float, x: FacTup, y: FacTup): FacTup = {*/
      /*  (rddAXPY(a,x._1,y._1),rddAXPY(a,x._2,y._2))*/
      /*}*/

      // form RDDs of (key,(x,p)) --- a "ray" with a point and a direction
      val userRay: RDD[(Int, (FactorBlock,FactorBlock))] = userFac.join(userDirec).cache()
      val itemRay: RDD[(Int, (FactorBlock,FactorBlock))] = itemFac.join(itemDirec).cache()

      def newPoint(a:Float, x: (FactorBlock,FactorBlock) ): FactorBlock = blockAXPY(a,x._2,x._1)
      def newUser(a: Float): FactorRDD = userRay.mapValues{ x => newPoint(a,x) }
      def newItem(a: Float): FactorRDD = itemRay.mapValues{ x => newPoint(a,x) }

      def axpy(a: Float): FacTup = {
        val u = newUser(a);
        val i = newItem(a);
        (u,i)
      }

      val (userGrad,itemGrad) = computeGradient(userFac,itemFac)
      val gradientProdDirec: Float = rddDOT(userGrad,userDirec) + rddDOT(itemGrad,itemDirec)
      userGrad.unpersist()
      itemGrad.unpersist()

      val alpha = linesearch(
        costFunc,
        axpy,
        (userFac,itemFac),
        (userDirec,itemDirec),
        reduceFrac,
        initStep,
        gradFrac * gradientProdDirec,
        maxIters
      )
      userRay.unpersist()
      itemRay.unpersist()

      alpha
    }
        
    /* initialize the factor vectors for:
     * user, item  -- user/item factors
     * grad -- gradient of f(U,M)
     * direc -- search direction
     *
     * the suffix _old signifies the vector from the previous iteration
     * the suffix _pc signifies an ALS-preconditioned vector
     */

    // generate initial factor vectors
    val seedGen = new XORShiftRandom(seed)
    var users: FactorRDD = initialize(userInBlocks, rank, seedGen.nextLong()).cache()
    var items: FactorRDD = initialize(itemInBlocks, rank, seedGen.nextLong()).cache()
    logStdout("PNCG: -1:" + (users.count + items.count));

    var users_pc: FactorRDD = preconditionUsers(items).cache()
    var items_pc: FactorRDD = preconditionItems(users_pc).cache()

    // compute preconditioned gradients; g = x - x_pc
    var gradUser: FactorRDD = rddAXPY(-1.0f,users_pc,users).cache()
    var gradItem: FactorRDD = rddAXPY(-1.0f,items_pc,items).cache()

    // initialize variables for the previous iteration's gradients
    var gradUser_old: FactorRDD = gradUser.cache()
    var gradItem_old: FactorRDD = gradItem.cache()

    // initial search direction to -gradient (steepest descent direction)
    var direcUser: FactorRDD = gradUser.mapValues{x => blockSCAL(x,-1.0f)}.cache()
    var direcItem: FactorRDD = gradItem.mapValues{x => blockSCAL(x,-1.0f)}.cache()

    // compute g^T * g
    // make variable for the actual gradient ---bad naming, I know. Have to fix this
    val (gu,gi) = computeGradient(users,items)
    var gradTgrad = rddDOT(gu,gradUser) + rddDOT(gi,gradItem);
    var gradTgrad_old = 0.0f;

    val alpha_max: Float = 10.0f
    var beta_pncg: Float = gradTgrad
    var alpha_pncg: Float = alpha_max
    val checkpointInterval: Int = 15 

    logStdout("PNCG: 0:" + gradTgrad);
    var iter: Int = 1
    for (iter <- 1 until maxIter+1) 
    {
      // store old preconditioned gradient vectors for computing \beta
      gradTgrad_old = gradTgrad
      gradUser_old = gradUser.cache()
      gradItem_old = gradItem.cache()

      //compute alpha from linesearch()
      val alpha0 = {
        if (alpha_pncg > 1e-4)
          2*alpha_pncg
        else
          alpha_max
      }
      alpha_pncg = computeAlpha(
        users,
        items,
        direcUser,
        direcItem,
        alpha0,
        0.5f,
        0.5f,
        10
      )
      // x_{k+1} = x_k + \alpha * p_k

      users = rddAXPY(alpha_pncg, direcUser, users).cache()
      items = rddAXPY(alpha_pncg, direcItem, items).cache()

      if (sc.checkpointDir.isDefined && (iter % checkpointInterval == 0))
      {
        users.checkpoint()
        items.checkpoint()
        items.count()
        users.count()
      }

      // precondition x with ALS
      // \bar{x} = P * \x_{k+1}
      users_pc = preconditionUsers(items).cache()
      items_pc = preconditionItems(users_pc).cache()

      // compute the preconditioned gradient
      // g = x_{k+1} - \bar{x} 
      gradUser = rddAXPY(-1.0f,users_pc,users).cache() // x - x_pc
      gradItem = rddAXPY(-1.0f,items_pc,items).cache() // x - x_pc

      // compute beta 
      //======================================================
      //FR
      /*gradTgrad = (rddNORMSQR(gradUser) + rddNORMSQR(gradItem));*/
      /*beta_pncg = gradTgrad / gradTgrad_old*/

      // PR
      val (gu,gi) = computeGradient(users,items)
      gradTgrad = rddDOT(gu,gradUser) + rddDOT(gi,gradItem);
      beta_pncg = {
        if (gradTgrad_old > 0.0f)
          (gradTgrad - (rddDOT(gu,gradUser_old) + rddDOT(gradItem,gradItem_old)) ) / gradTgrad_old
        else
        {
          logStdout("PNCG: Restarting in steepest descent direction; beta = 0")
          0f
        }
      }

      // p_{k+1} = -g + \beta * p_k
      direcUser = rddAXPBY(-1.0f,gradUser,beta_pncg,direcUser).cache()
      direcItem = rddAXPBY(-1.0f,gradItem,beta_pncg,direcItem).cache()

      if (sc.checkpointDir.isDefined && (iter % checkpointInterval == 0))
      {
        direcUser.checkpoint()
        direcItem.checkpoint()
        direcUser.count()
        direcItem.count()
      }

      //materialize RDDs
      gradUser_old.unpersist()
      gradItem_old.unpersist()
      gu.unpersist()
      gi.unpersist()
      logStdout("PNCG: "+ iter+ ":" + (direcUser.count + direcItem.count ) )
    }

    val userIdAndFactors = userInBlocks
      .mapValues(_.srcIds)
      .join(users)
      .mapPartitions({ p =>
        p.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
      // and userFactors.
      }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors = itemInBlocks
      .mapValues(_.srcIds)
      .join(items)
      .mapPartitions({ p =>
        p.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      items.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }

  /**
   * Implementation of the ALS algorithm.
   */
  def train[ID: ClassTag]( // scalastyle:ignore
      ratings: RDD[Rating[ID]],
      rank: Int = 10,
      numUserBlocks: Int = 10,
      numItemBlocks: Int = 10,
      maxIter: Int = 10,
      regParam: Double = 1.0,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      nonnegative: Boolean = false,
      intermediateRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      finalRDDStorageLevel: StorageLevel = StorageLevel.MEMORY_AND_DISK,
      seed: Long = 0L)(
      implicit ord: Ordering[ID]): (RDD[(ID, Array[Float])], RDD[(ID, Array[Float])]) = {
    require(intermediateRDDStorageLevel != StorageLevel.NONE,
      "ALS is not designed to run without persisting intermediate RDDs.")
    val sc = ratings.sparkContext
    val userPart = new ALSPartitioner(numUserBlocks)
    val itemPart = new ALSPartitioner(numItemBlocks)
    val userLocalIndexEncoder = new LocalIndexEncoder(userPart.numPartitions)
    val itemLocalIndexEncoder = new LocalIndexEncoder(itemPart.numPartitions)
    val solver = if (nonnegative) new NNLSSolver else new CholeskySolver
    val blockRatings = partitionRatings(ratings, userPart, itemPart)
      .persist(intermediateRDDStorageLevel)
    val (userInBlocks, userOutBlocks, userCounts) =
      makeBlocks("user", blockRatings, userPart, itemPart, intermediateRDDStorageLevel)
    // materialize blockRatings and user blocks
    userCounts.count()
    userOutBlocks.count()
    val swappedBlockRatings = blockRatings.map {
      case ((userBlockId, itemBlockId), RatingBlock(userIds, itemIds, localRatings)) =>
        ((itemBlockId, userBlockId), RatingBlock(itemIds, userIds, localRatings))
    }
    val (itemInBlocks, itemOutBlocks, itemCounts) =
      makeBlocks("item", swappedBlockRatings, itemPart, userPart, intermediateRDDStorageLevel)
    // materialize item blocks
    itemCounts.count()
    itemOutBlocks.count()
    val seedGen = new XORShiftRandom(seed)
    var userFactors = initialize(userInBlocks, rank, seedGen.nextLong()).cache()
    var itemFactors = initialize(itemInBlocks, rank, seedGen.nextLong()).cache()

    logStdout("ALS:" + 0 +": "+ (userFactors.count + itemFactors.count) )
    if (implicitPrefs) {
      for (iter <- 1 to maxIter+1) {
        userFactors.setName(s"userFactors-$iter").persist(intermediateRDDStorageLevel)
        val previousItemFactors = itemFactors
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, implicitPrefs, alpha, solver)
        previousItemFactors.unpersist()
        if (sc.checkpointDir.isDefined && (iter % 15 == 0)) {
          itemFactors.checkpoint()
        }
        itemFactors.setName(s"itemFactors-$iter").persist(intermediateRDDStorageLevel)
        val previousUserFactors = userFactors
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, implicitPrefs, alpha, solver)
        previousUserFactors.unpersist()
      }
    } else {
      for (iter <- 1 until maxIter+1) {
        itemFactors = computeFactors(userFactors, userOutBlocks, itemInBlocks, rank, regParam,
          userLocalIndexEncoder, solver = solver).cache()
        if (sc.checkpointDir.isDefined && (iter % 15 == 0))
        {
          logStdout("Checkpointing at iter " + iter)
          itemFactors.checkpoint()
        }
        userFactors = computeFactors(itemFactors, itemOutBlocks, userInBlocks, rank, regParam,
          itemLocalIndexEncoder, solver = solver).cache()

        logStdout("ALS: " + iter + ":" + userFactors.count) 
      }
    }
    val userIdAndFactors = userInBlocks
      .mapValues(_.srcIds)
      .join(userFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      // Preserve the partitioning because IDs are consistent with the partitioners in userInBlocks
      // and userFactors.
      }, preservesPartitioning = true)
      .setName("userFactors")
      .persist(finalRDDStorageLevel)
    val itemIdAndFactors = itemInBlocks
      .mapValues(_.srcIds)
      .join(itemFactors)
      .mapPartitions({ items =>
        items.flatMap { case (_, (ids, factors)) =>
          ids.view.zip(factors)
        }
      }, preservesPartitioning = true)
      .setName("itemFactors")
      .persist(finalRDDStorageLevel)
    if (finalRDDStorageLevel != StorageLevel.NONE) {
      userIdAndFactors.count()
      itemFactors.unpersist()
      itemIdAndFactors.count()
      userInBlocks.unpersist()
      userOutBlocks.unpersist()
      itemInBlocks.unpersist()
      itemOutBlocks.unpersist()
      blockRatings.unpersist()
    }
    (userIdAndFactors, itemIdAndFactors)
  }

  /**
   * Factor block that stores factors (Array[Float]) in an Array.
   */
  private type FactorBlock = Array[Array[Float]]

  /**
   * Out-link block that stores, for each dst (item/user) block, which src (user/item) factors to
   * send. For example, outLinkBlock(0) contains the local indices (not the original src IDs) of the
   * src factors in this block to send to dst block 0.
   */
  private type OutBlock = Array[Array[Int]]

  /**
   * In-link block for computing src (user/item) factors. This includes the original src IDs
   * of the elements within this block as well as encoded dst (item/user) indices and corresponding
   * ratings. The dst indices are in the form of (blockId, localIndex), which are not the original
   * dst IDs. To compute src factors, we expect receiving dst factors that match the dst indices.
   * For example, if we have an in-link record
   *
   * {srcId: 0, dstBlockId: 2, dstLocalIndex: 3, rating: 5.0},
   *
   * and assume that the dst factors are stored as dstFactors: Map[Int, Array[Array[Float]]], which
   * is a blockId to dst factors map, the corresponding dst factor of the record is dstFactor(2)(3).
   *
   * We use a CSC-like (compressed sparse column) format to store the in-link information. So we can
   * compute src factors one after another using only one normal equation instance.
   *
   * @param srcIds src ids (ordered)
   * @param dstPtrs dst pointers. Elements in range [dstPtrs(i), dstPtrs(i+1)) of dst indices and
   *                ratings are associated with srcIds(i).
   * @param dstEncodedIndices encoded dst indices
   * @param ratings ratings
   *
   * @see [[LocalIndexEncoder]]
   */
  private[recommendation] case class InBlock[@specialized(Int, Long) ID: ClassTag](
      srcIds: Array[ID],
      dstPtrs: Array[Int],
      dstEncodedIndices: Array[Int],
      ratings: Array[Float]) {
    /** Size of the block. */
    def size: Int = ratings.length
    require(dstEncodedIndices.length == size)
    require(dstPtrs.length == srcIds.length + 1)
  }

  /**
   * Initializes factors randomly given the in-link blocks.
   *
   * @param inBlocks in-link blocks
   * @param rank rank
   * @return initialized factor blocks
   */
  private def initialize[ID](
      inBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      seed: Long): RDD[(Int, FactorBlock)] = {
    // Choose a unit vector uniformly at random from the unit sphere, but from the
    // "first quadrant" where all elements are nonnegative. This can be done by choosing
    // elements distributed as Normal(0,1) and taking the absolute value, and then normalizing.
    // This appears to create factorizations that have a slightly better reconstruction
    // (<1%) compared picking elements uniformly at random in [0,1].
    inBlocks.map { case (srcBlockId, inBlock) =>
      val random = new XORShiftRandom(byteswap64(seed ^ srcBlockId))
      val factors = Array.fill(inBlock.srcIds.length) {

        // setting this to be all ones, just for now. ---REPLACE THIS
        /*val factor = Array.fill(rank)(1.0f)*/

        val factor = Array.fill(rank)(random.nextGaussian().toFloat)
        val nrm = blas.snrm2(rank, factor, 1)
        blas.sscal(rank, 1.0f / nrm, factor, 1)
        factor
      }
      (srcBlockId, factors)
    }
  }

  /**
   * A rating block that contains src IDs, dst IDs, and ratings, stored in primitive arrays.
   */
  private[recommendation] case class RatingBlock[@specialized(Int, Long) ID: ClassTag](
      srcIds: Array[ID],
      dstIds: Array[ID],
      ratings: Array[Float]) {
    /** Size of the block. */
    def size: Int = srcIds.length
    require(dstIds.length == srcIds.length)
    require(ratings.length == srcIds.length)
  }

  /**
   * Builder for [[RatingBlock]]. [[mutable.ArrayBuilder]] is used to avoid boxing/unboxing.
   */
  private[recommendation] class RatingBlockBuilder[@specialized(Int, Long) ID: ClassTag]
    extends Serializable {

    private val srcIds = mutable.ArrayBuilder.make[ID]
    private val dstIds = mutable.ArrayBuilder.make[ID]
    private val ratings = mutable.ArrayBuilder.make[Float]
    var size = 0

    /** Adds a rating. */
    def add(r: Rating[ID]): this.type = {
      size += 1
      srcIds += r.user
      dstIds += r.item
      ratings += r.rating
      this
    }

    /** Merges another [[RatingBlockBuilder]]. */
    def merge(other: RatingBlock[ID]): this.type = {
      size += other.srcIds.length
      srcIds ++= other.srcIds
      dstIds ++= other.dstIds
      ratings ++= other.ratings
      this
    }

    /** Builds a [[RatingBlock]]. */
    def build(): RatingBlock[ID] = {
      RatingBlock[ID](srcIds.result(), dstIds.result(), ratings.result())
    }
  }

  /**
   * Partitions raw ratings into blocks.
   *
   * @param ratings raw ratings
   * @param srcPart partitioner for src IDs
   * @param dstPart partitioner for dst IDs
   *
   * @return an RDD of rating blocks in the form of ((srcBlockId, dstBlockId), ratingBlock)
   */
  private def partitionRatings[ID: ClassTag](
      ratings: RDD[Rating[ID]],
      srcPart: Partitioner,
      dstPart: Partitioner): RDD[((Int, Int), RatingBlock[ID])] = {

     /* The implementation produces the same result as the following but generates fewer objects.

     ratings.map { r =>
       ((srcPart.getPartition(r.user), dstPart.getPartition(r.item)), r)
     }.aggregateByKey(new RatingBlockBuilder)(
         seqOp = (b, r) => b.add(r),
         combOp = (b0, b1) => b0.merge(b1.build()))
       .mapValues(_.build())
     */

    val numPartitions = srcPart.numPartitions * dstPart.numPartitions
    ratings.mapPartitions { iter =>
      val builders = Array.fill(numPartitions)(new RatingBlockBuilder[ID])
      iter.flatMap { r =>
        val srcBlockId = srcPart.getPartition(r.user)
        val dstBlockId = dstPart.getPartition(r.item)
        val idx = srcBlockId + srcPart.numPartitions * dstBlockId
        val builder = builders(idx)
        builder.add(r)
        if (builder.size >= 2048) { // 2048 * (3 * 4) = 24k
          builders(idx) = new RatingBlockBuilder
          Iterator.single(((srcBlockId, dstBlockId), builder.build()))
        } else {
          Iterator.empty
        }
      } ++ {
        builders.view.zipWithIndex.filter(_._1.size > 0).map { case (block, idx) =>
          val srcBlockId = idx % srcPart.numPartitions
          val dstBlockId = idx / srcPart.numPartitions
          ((srcBlockId, dstBlockId), block.build())
        }
      }
    }.groupByKey().mapValues { blocks =>
      val builder = new RatingBlockBuilder[ID]
      blocks.foreach(builder.merge)
      builder.build()
    }.setName("ratingBlocks")
  }

  /**
   * Builder for uncompressed in-blocks of (srcId, dstEncodedIndex, rating) tuples.
   * @param encoder encoder for dst indices
   */
  private[recommendation] class UncompressedInBlockBuilder[@specialized(Int, Long) ID: ClassTag](
      encoder: LocalIndexEncoder)(
      implicit ord: Ordering[ID]) {

    private val srcIds = mutable.ArrayBuilder.make[ID]
    private val dstEncodedIndices = mutable.ArrayBuilder.make[Int]
    private val ratings = mutable.ArrayBuilder.make[Float]

    /**
     * Adds a dst block of (srcId, dstLocalIndex, rating) tuples.
     *
     * @param dstBlockId dst block ID
     * @param srcIds original src IDs
     * @param dstLocalIndices dst local indices
     * @param ratings ratings
     */
    def add(
        dstBlockId: Int,
        srcIds: Array[ID],
        dstLocalIndices: Array[Int],
        ratings: Array[Float]): this.type = {
      val sz = srcIds.length
      require(dstLocalIndices.length == sz)
      require(ratings.length == sz)
      this.srcIds ++= srcIds
      this.ratings ++= ratings
      var j = 0
      while (j < sz) {
        this.dstEncodedIndices += encoder.encode(dstBlockId, dstLocalIndices(j))
        j += 1
      }
      this
    }

    /** Builds a [[UncompressedInBlock]]. */
    def build(): UncompressedInBlock[ID] = {
      new UncompressedInBlock(srcIds.result(), dstEncodedIndices.result(), ratings.result())
    }
  }

  /**
   * A block of (srcId, dstEncodedIndex, rating) tuples stored in primitive arrays.
   */
  private[recommendation] class UncompressedInBlock[@specialized(Int, Long) ID: ClassTag](
      val srcIds: Array[ID],
      val dstEncodedIndices: Array[Int],
      val ratings: Array[Float])(
      implicit ord: Ordering[ID]) {

    /** Size the of block. */
    def length: Int = srcIds.length


    /** Count the number of ratings per user/item 
     */
    def countRatings(): Array[Float] = {
      val len = length
      assert(len > 0, "Empty in-link block should not exist.")
      sort()
      val dstCountsBuilder = mutable.ArrayBuilder.make[Float]
      var preSrcId = srcIds(0)
      var curCount = 1
      var i = 1
      var j = 0
      while (i < len) {
        val srcId = srcIds(i)
        if (srcId != preSrcId) {
          dstCountsBuilder += curCount
          preSrcId = srcId
          j += 1
          curCount = 0
        }
        curCount += 1
        i += 1
      }
      dstCountsBuilder += curCount

      dstCountsBuilder.result()
    }
    /**
     * Compresses the block into an [[InBlock]]. The algorithm is the same as converting a
     * sparse matrix from coordinate list (COO) format into compressed sparse column (CSC) format.
     * Sorting is done using Spark's built-in Timsort to avoid generating too many objects.
     */
    def compress(): InBlock[ID] = {
      val sz = length
      assert(sz > 0, "Empty in-link block should not exist.")
      sort()
      val uniqueSrcIdsBuilder = mutable.ArrayBuilder.make[ID]
      val dstCountsBuilder = mutable.ArrayBuilder.make[Int]
      var preSrcId = srcIds(0)
      uniqueSrcIdsBuilder += preSrcId
      var curCount = 1
      var i = 1
      var j = 0
      while (i < sz) {
        val srcId = srcIds(i)
        if (srcId != preSrcId) {
          uniqueSrcIdsBuilder += srcId
          dstCountsBuilder += curCount
          preSrcId = srcId
          j += 1
          curCount = 0
        }
        curCount += 1
        i += 1
      }
      dstCountsBuilder += curCount
      val uniqueSrcIds = uniqueSrcIdsBuilder.result()
      val numUniqueSrdIds = uniqueSrcIds.length
      val dstCounts = dstCountsBuilder.result()
      val dstPtrs = new Array[Int](numUniqueSrdIds + 1)
      var sum = 0
      i = 0
      while (i < numUniqueSrdIds) {
        sum += dstCounts(i)
        i += 1
        dstPtrs(i) = sum
      }
      InBlock(uniqueSrcIds, dstPtrs, dstEncodedIndices, ratings)
    }

    private def sort(): Unit = {
      val sz = length
      // Since there might be interleaved log messages, we insert a unique id for easy pairing.
      val sortId = Utils.random.nextInt()
      logDebug(s"Start sorting an uncompressed in-block of size $sz. (sortId = $sortId)")
      val start = System.nanoTime()
      val sorter = new Sorter(new UncompressedInBlockSort[ID])
      sorter.sort(this, 0, length, Ordering[KeyWrapper[ID]])
      val duration = (System.nanoTime() - start) / 1e9
      logDebug(s"Sorting took $duration seconds. (sortId = $sortId)")
    }
  }

  /**
   * A wrapper that holds a primitive key.
   *
   * @see [[UncompressedInBlockSort]]
   */
  private class KeyWrapper[@specialized(Int, Long) ID: ClassTag](
      implicit ord: Ordering[ID]) extends Ordered[KeyWrapper[ID]] {

    var key: ID = _

    override def compare(that: KeyWrapper[ID]): Int = {
      ord.compare(key, that.key)
    }

    def setKey(key: ID): this.type = {
      this.key = key
      this
    }
  }

  /**
   * [[SortDataFormat]] of [[UncompressedInBlock]] used by [[Sorter]].
   */
  private class UncompressedInBlockSort[@specialized(Int, Long) ID: ClassTag](
      implicit ord: Ordering[ID])
    extends SortDataFormat[KeyWrapper[ID], UncompressedInBlock[ID]] {

    override def newKey(): KeyWrapper[ID] = new KeyWrapper()

    override def getKey(
        data: UncompressedInBlock[ID],
        pos: Int,
        reuse: KeyWrapper[ID]): KeyWrapper[ID] = {
      if (reuse == null) {
        new KeyWrapper().setKey(data.srcIds(pos))
      } else {
        reuse.setKey(data.srcIds(pos))
      }
    }

    override def getKey(
        data: UncompressedInBlock[ID],
        pos: Int): KeyWrapper[ID] = {
      getKey(data, pos, null)
    }

    private def swapElements[@specialized(Int, Float) T](
        data: Array[T],
        pos0: Int,
        pos1: Int): Unit = {
      val tmp = data(pos0)
      data(pos0) = data(pos1)
      data(pos1) = tmp
    }

    override def swap(data: UncompressedInBlock[ID], pos0: Int, pos1: Int): Unit = {
      swapElements(data.srcIds, pos0, pos1)
      swapElements(data.dstEncodedIndices, pos0, pos1)
      swapElements(data.ratings, pos0, pos1)
    }

    override def copyRange(
        src: UncompressedInBlock[ID],
        srcPos: Int,
        dst: UncompressedInBlock[ID],
        dstPos: Int,
        length: Int): Unit = {
      System.arraycopy(src.srcIds, srcPos, dst.srcIds, dstPos, length)
      System.arraycopy(src.dstEncodedIndices, srcPos, dst.dstEncodedIndices, dstPos, length)
      System.arraycopy(src.ratings, srcPos, dst.ratings, dstPos, length)
    }

    override def allocate(length: Int): UncompressedInBlock[ID] = {
      new UncompressedInBlock(
        new Array[ID](length), new Array[Int](length), new Array[Float](length))
    }

    override def copyElement(
        src: UncompressedInBlock[ID],
        srcPos: Int,
        dst: UncompressedInBlock[ID],
        dstPos: Int): Unit = {
      dst.srcIds(dstPos) = src.srcIds(srcPos)
      dst.dstEncodedIndices(dstPos) = src.dstEncodedIndices(srcPos)
      dst.ratings(dstPos) = src.ratings(srcPos)
    }
  }

  /**
   * Creates in-blocks and out-blocks from rating blocks.
   * @param prefix prefix for in/out-block names
   * @param ratingBlocks rating blocks
   * @param srcPart partitioner for src IDs
   * @param dstPart partitioner for dst IDs
   * @return (in-blocks, out-blocks)
   */
  private def makeBlocks[ID: ClassTag](
      prefix: String,
      ratingBlocks: RDD[((Int, Int), RatingBlock[ID])],
      srcPart: Partitioner,
      dstPart: Partitioner,
      storageLevel: StorageLevel)(
      implicit srcOrd: Ordering[ID]): (RDD[(Int, InBlock[ID])], RDD[(Int, OutBlock)], RDD[(Int, Array[Float])]) = {

    /**
     * compute the local destination indices for each index i as
     * i_local = mod(i,N), where N is the nu
     */ 
    def computeLocalIndices(dstIds: Array[ID]): Array[Int] = {

      val start = System.nanoTime()
      val dstIdSet = new OpenHashSet[ID](1 << 20)
      dstIds.foreach(dstIdSet.add)

      // The implementation is a faster version of
      // val dstIdToLocalIndex = dstIds.toSet.toSeq.sorted.zipWithIndex.toMap
      val sortedDstIds = new Array[ID](dstIdSet.size)
      var i = 0
      var pos = dstIdSet.nextPos(0)
      while (pos != -1) {
        sortedDstIds(i) = dstIdSet.getValue(pos)
        pos = dstIdSet.nextPos(pos + 1)
        i += 1
      }
      assert(i == dstIdSet.size)
      Sorting.quickSort(sortedDstIds)
      val len = sortedDstIds.length
      val dstIdToLocalIndex = new OpenHashMap[ID, Int](len)

      i = 0
      while (i < len) {
        dstIdToLocalIndex.update(sortedDstIds(i), i)
        i += 1
      }
      logDebug("Converting to local indices took " 
        + (System.nanoTime() - start) / 1e9 
        + " seconds.")

      dstIds.map(dstIdToLocalIndex.apply)
    }

    type UncompressedCols = (Int, Array[ID], Array[Int], Array[Float])

    def toUncompressedCols(key: (Int,Int), block: RatingBlock[ID]): (Int, UncompressedCols) = {
      val localBlockInds: Array[Int] = computeLocalIndices(block.dstIds)
      (key._1, (key._2, block.srcIds, localBlockInds, block.ratings) ) 
    }

    def toUncompressedSparseCols(iter: Iterable[UncompressedCols]): UncompressedInBlock[ID] = {
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val builder = new UncompressedInBlockBuilder[ID](encoder)
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
      }
      builder.build()
    }

    def toCounts(iter: Iterable[UncompressedCols]): Array[Float] = {
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val builder = new UncompressedInBlockBuilder[ID](encoder)
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
      }
      builder.build().countRatings()
    }

    def toCompressedSparseCols(iter: Iterable[UncompressedCols]): InBlock[ID] = {
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val builder = new UncompressedInBlockBuilder[ID](encoder)
      iter.foreach { case (dstBlockId, srcIds, dstLocalIndices, ratings) =>
        builder.add(dstBlockId, srcIds, dstLocalIndices, ratings)
      }
      builder.build().compress()
    }

    def toOutLinkArray(block: InBlock[ID]): OutBlock = { 
      val encoder = new LocalIndexEncoder(dstPart.numPartitions)
      val activeIds = Array.fill(dstPart.numPartitions)(mutable.ArrayBuilder.make[Int])
      var i = 0
      val seen = new Array[Boolean](dstPart.numPartitions)
      while (i < block.srcIds.length) {
        var j = block.dstPtrs(i)
        ju.Arrays.fill(seen, false)
        while (j < block.dstPtrs(i + 1)) {
          val dstBlockId = encoder.blockId(block.dstEncodedIndices(j))
          if (!seen(dstBlockId)) {
            activeIds(dstBlockId) += i // add the local index in this out-block
            seen(dstBlockId) = true
          }
          j += 1
        }
        i += 1
      }
      activeIds.map { x =>
        x.result()
      }
    }

    val counts: RDD[(Int, Array[Float])] = ratingBlocks
      .map{ case(key,block) => toUncompressedCols(key,block) } //(BlockId, (Int, Array[ID], Array[Int], Array[Float]) )
      .groupByKey(new ALSPartitioner(srcPart.numPartitions))
      .mapValues(toCounts)
      .setName(prefix + "RatingsCounts")
      .persist(storageLevel)

    val inBlocks = ratingBlocks
      .map{ case(key,block) => toUncompressedCols(key,block) } //(BlockId, (Int, Array[ID], Array[Int], Array[Float]) )
      .groupByKey(new ALSPartitioner(srcPart.numPartitions))
      .mapValues(toCompressedSparseCols)
      .setName(prefix + "InBlocks")
      .persist(storageLevel)

    val outBlocks = inBlocks
      .mapValues(toOutLinkArray)
      .setName(prefix + "OutBlocks")
      .persist(storageLevel)

    (inBlocks, outBlocks, counts)
  }

  /**
   * Evaluate the gradient function f(U,M), as in \cite{zhou2008largescale}
   *
   * Comments are written assuming the gradient WRT users is being calculated.
   * For, e.g. the ith user u_i:
   *  1/2 * df(u_i)/du_i = (M_i * M^T_i + \lambda * n_{u_i} )*u_i - M_{u_i}*R^T_{u_i}
   *
   * @param srcFactorBlocks src factors; the item factors, m_i
   * @param currentFactorBlocks current user factors, u_i
   * @param srcOutBlocks src out-blocks
   * @param dstInBlocks dst in-blocks
   * @param rank rank
   * @param regParam regularization constant
   * @param srcEncoder encoder for src local indices
   * @param implicitPrefs whether to use implicit preference
   * @param alpha the alpha constant in the implicit preference formulation
   * @param solver solver for least squares problems
   *
   * @return grad gradient vector 
   */
  private def evalGradient[ID](
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      currentFactorBlocks: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      regParam: Double,
      srcEncoder: LocalIndexEncoder,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0
      ): RDD[(Int, FactorBlock)] = 
  {
    val numSrcBlocks = srcFactorBlocks.partitions.length

    def filterFactorsToSend(
        srcBlockId: Int, 
        tup: (OutBlock, FactorBlock)) = 
    {

      val block = tup._1
      val factors = tup._2
      block
        .view
        .zipWithIndex
        .map{ case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(factors(_))))
        }
    }

    def computeGradientBlock(
        block: InBlock[ID],  //{m_j}
        factorList: Iterable[(Int,FactorBlock)],
        current: FactorBlock 
        ): FactorBlock = 
    {
      val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
      factorList.foreach { case (srcBlockId, vec) =>
        sortedSrcFactors(srcBlockId) = vec
      }
      val len = block.srcIds.length

      // initialize array of gradient vectors
      val grad: Array[Array[Float]] = Array.fill(len)(Array.fill[Float](rank)(0f))

      var i = 0

      // loop over all users {u_i}
      while (i < len) 
      {
        // loop over all input {m_j} in block
        var j = block.dstPtrs(i)
        var num_factors = 0
        while (j < block.dstPtrs(i + 1)) 
        {
          val encoded = block.dstEncodedIndices(j)
          val blockId = srcEncoder.blockId(encoded)
          val localIndex = srcEncoder.localIndex(encoded)
          val srcFactor = sortedSrcFactors(blockId)(localIndex) //m_
          val rating = block.ratings(j)
          val a = blas.sdot(rank,current(i),1,srcFactor,1) - rating

          // y := a*x + y 
          blas.saxpy(rank,a,srcFactor ,1,grad(i),1)
          j += 1
          num_factors += 1
        }
        // add \lambda * n * u_i
        blas.saxpy(rank,regParam.toFloat*num_factors,current(i),1,grad(i),1)
        blas.sscal(rank,2.0f,grad(i),1)
        i += 1
      }
      grad
    }

    val srcOut: RDD[(Int, Iterable[(Int,FactorBlock)]) ] = 
      srcOutBlocks
      .join(srcFactorBlocks)
      .flatMap{case (id,tuple) => filterFactorsToSend(id,tuple)}
      .groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))

    val gradient: RDD[(Int, FactorBlock)] = dstInBlocks
      .join(srcOut)
      .join(currentFactorBlocks)
      .mapValues{case ((block,factorTuple),fac) => computeGradientBlock(block,factorTuple,fac)}
      
      /*.cogroup(srcOut,currentFactorBlocks)*/
      //use .head since cogroup has produced Iterables
      /*.mapValues{case (block,factorTuple,fac) => */
      /*  computeGradientBlock(block.head,factorTuple.head,fac.head)*/
      /*}*/
    gradient
  }

  /**
   * Evaluate the Tikhonov normalization for f(U,M)
   *
   * @param factors Array of Array[Float] factors; the item factors, m_i
   * @param counts the number of ratings associated with each factor, Array[Int] 
   * @param rank the size of a single factor vector
   *
   */
  private def evalTikhonovNorm(
      factors: RDD[(Int, FactorBlock)],
      counts: RDD[(Int, Array[Float])],
      rank: Int,
      lambda: Double
      ): Float = 
  {
    def evalBlockNorms(block: FactorBlock): Array[Float] = 
    {
      val numFactors: Int = block.length
      val result: Array[Float] = new Array[Float](numFactors)
      var j: Int = 0
      while (j < numFactors) {
        result(j)= blas.sdot(rank,block(j),1,block(j),1)
        /*logStdout("dot" + j + " = " + result(j))*/
        j += 1
      }
      result
    }
    def scaleByNumRatings(factorNorms: Array[Float], numRatings: Array[Float]): Float =
    {
      val numFactors: Int = factorNorms.length
      var result: Float = 0f
      var j: Int = 0
      /*while (j < numFactors) {*/
      /*  result += factorNorms(j) * numRatings(j)*/
      /*  /*logStdout("scale" + j + " : " + result)*/*/
      /*  j += 1*/
      /*}*/
      /*result*/
      blas.sdot(numFactors,factorNorms,1,numRatings,1)
    }

    val factorNorm: Float = factors
      .mapValues{evalBlockNorms}
      .join(counts)
      .map{case (key,(f,n)) => scaleByNumRatings(f,n)}
      .reduce(_ + _)

    lambda.toFloat * factorNorm
  }

  /**
   * Compute the Frobenius norm part of the cost function for the current set of factors 
   *
   * @param srcFactorBlocks src factors
   * @param currentFactorBlocks current user factors, u_i
   * @param srcOutBlocks src out-blocks
   * @param dstInBlocks dst in-blocks
   * @param rank rank
   * @param regParam regularization constant
   * @param srcEncoder encoder for src local indices
   * @param implicitPrefs whether to use implicit preference
   * @param alpha the alpha constant in the implicit preference formulation
   * @param solver solver for least squares problems
   *
   * @return dst factors
   */
  private def evalFrobeniusCost[ID](
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      currentFactorBlocks: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      regParam: Double,
      srcEncoder: LocalIndexEncoder,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0
      ): Float = 
    {

    val numSrcBlocks = srcFactorBlocks.partitions.length
    val YtY = 
      if (implicitPrefs) 
        Some(computeYtY(srcFactorBlocks, rank)) 
      else 
        None

    /*type BlockFacTuple = (OutBlock,FactorBlock)*/
    def filterFactorsToSend(
        srcBlockId: Int, 
        tup: (OutBlock, FactorBlock)) = {

      val block = tup._1
      val factors = tup._2
      block
        .view
        .zipWithIndex
        .map{ case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(factors(_))))
        }
    }

    def computeSquaredError(
        block: InBlock[ID], 
        /*factorList: Iterable[(Int,FactorBlock)],*/
        factorList: Iterable[(Int,FactorBlock)],
        current: FactorBlock 
        ): Float = 
    {

      val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
      factorList.foreach { case (srcBlockId, vec) =>
        sortedSrcFactors(srcBlockId) = vec
      }
      val len = block.srcIds.length
      /*logStdout("block.srcIds.length = " + len)*/
      /*logStdout("current.length = " + current.length)*/
      var j = 0
      var sumErrs: Double = 0
      while (j < len) 
      {
        var i = block.dstPtrs(j)
        while (i < block.dstPtrs(j + 1)) {
          val encoded = block.dstEncodedIndices(i)
          val blockId = srcEncoder.blockId(encoded)
          val localIndex = srcEncoder.localIndex(encoded)
          val srcFactor = sortedSrcFactors(blockId)(localIndex)
          val rating = block.ratings(i)
          val diff = blas.sdot(rank,current(j),1,srcFactor,1) - rating
          sumErrs += math.pow(diff.toDouble, 2)
          i += 1
        }
        j += 1
      }
      sumErrs.toFloat
    }

    val srcOut: RDD[(Int, Iterable[(Int,FactorBlock)]) ] = srcOutBlocks
      .join(srcFactorBlocks)
      .flatMap{case (id,tuple) => filterFactorsToSend(id,tuple)}
      .groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))

    val result: Double = dstInBlocks
      .join(srcOut)
      .join(currentFactorBlocks)
      /*.map{(key,((block,factorTuple),fac) ) => computeSquaredError(block,factorTuple,fac)}*/
      /*.cogroup(srcOut,currentFactorBlocks)*/
      .mapValues{case ( (block,factorTuple),fac) => 
        computeSquaredError(block,factorTuple,fac)
      }
      .values
      .reduce(_+_)

    result.toFloat
  }


  /**
   * Compute dst factors by constructing and solving least square problems.
   *
   * @param srcFactorBlocks src factors
   * @param srcOutBlocks src out-blocks
   * @param dstInBlocks dst in-blocks
   * @param rank rank
   * @param regParam regularization constant
   * @param srcEncoder encoder for src local indices
   * @param implicitPrefs whether to use implicit preference
   * @param alpha the alpha constant in the implicit preference formulation
   * @param solver solver for least squares problems
   *
   * @return dst factors
   */
  private def computeFactors[ID](
      srcFactorBlocks: RDD[(Int, FactorBlock)],
      srcOutBlocks: RDD[(Int, OutBlock)],
      dstInBlocks: RDD[(Int, InBlock[ID])],
      rank: Int,
      regParam: Double,
      srcEncoder: LocalIndexEncoder,
      implicitPrefs: Boolean = false,
      alpha: Double = 1.0,
      solver: LeastSquaresNESolver): RDD[(Int, FactorBlock)] = 
    {

    val numSrcBlocks = srcFactorBlocks.partitions.length
    val YtY = 
      if (implicitPrefs) 
        Some(computeYtY(srcFactorBlocks, rank)) 
      else 
        None

    /*type BlockFacTuple = (OutBlock,FactorBlock)*/
    def filterFactorsToSend(
        srcBlockId: Int, 
        tup: (OutBlock, FactorBlock)) = {

      val block = tup._1
      val factors = tup._2
      block
        .view
        .zipWithIndex
        .map{ case (activeIndices, dstBlockId) =>
          (dstBlockId, (srcBlockId, activeIndices.map(factors(_))))
        }
    }

    def solveNormalEqn(
        block: InBlock[ID], 
        factorList: Iterable[(Int,FactorBlock)] ): FactorBlock = {

      val sortedSrcFactors = new Array[FactorBlock](numSrcBlocks)
      factorList.foreach { case (srcBlockId, vec) =>
        sortedSrcFactors(srcBlockId) = vec
      }
      val len = block.srcIds.length
      val dstFactors = new Array[Array[Float]](len)
      var j = 0
      val normEqn = new NormalEquation(rank)
      while (j < len) {
        normEqn.reset()
        if (implicitPrefs) {
          normEqn.merge(YtY.get)
        }
        var i = block.dstPtrs(j)
        while (i < block.dstPtrs(j + 1)) {
          val encoded = block.dstEncodedIndices(i)
          val blockId = srcEncoder.blockId(encoded)
          val localIndex = srcEncoder.localIndex(encoded)
          val srcFactor = sortedSrcFactors(blockId)(localIndex)
          val rating = block.ratings(i)
          if (implicitPrefs) {
            normEqn.addImplicit(srcFactor, rating, alpha)
          } else {
            normEqn.add(srcFactor, rating)
          }
          i += 1
        }
        dstFactors(j) = solver.solve(normEqn, regParam)
        j += 1
      }
      dstFactors
    }

    val srcOut: RDD[(Int, Iterable[(Int,FactorBlock)]) ] = 
      srcOutBlocks
      .join(srcFactorBlocks)
      .flatMap{case (id,tuple) => filterFactorsToSend(id,tuple)}
      .groupByKey(new ALSPartitioner(dstInBlocks.partitions.length))

    val newFactors: RDD[(Int, FactorBlock)] = 
      dstInBlocks
      .join(srcOut)
      .mapValues{case (block, factorTuple) => solveNormalEqn(block,factorTuple)}

    newFactors
  }

  /**
   * Computes the Gramian matrix of user or item factors, which is only used in implicit preference.
   * Caching of the input factors is handled in [[ALS#train]].
   */
  private def computeYtY(factorBlocks: RDD[(Int, FactorBlock)], rank: Int): NormalEquation = {
    factorBlocks.values.aggregate(new NormalEquation(rank))(
      seqOp = (ne, factors) => {
        factors.foreach(ne.add(_, 0.0f))
        ne
      },
      combOp = (ne1, ne2) => ne1.merge(ne2))
  }

  /**
   * Encoder for storing (blockId, localIndex) into a single integer.
   *
   * We use the leading bits (including the sign bit) to store the block id and the rest to store
   * the local index. This is based on the assumption that users/items are approximately evenly
   * partitioned. With this assumption, we should be able to encode two billion distinct values.
   *
   * @param numBlocks number of blocks
   */
  private[recommendation] class LocalIndexEncoder(numBlocks: Int) extends Serializable {

    require(numBlocks > 0, s"numBlocks must be positive but found $numBlocks.")

    private[this] final val numLocalIndexBits =
      math.min(java.lang.Integer.numberOfLeadingZeros(numBlocks - 1), 31)
    private[this] final val localIndexMask = (1 << numLocalIndexBits) - 1

    /** Encodes a (blockId, localIndex) into a single integer. */
    def encode(blockId: Int, localIndex: Int): Int = {
      require(blockId < numBlocks)
      require((localIndex & ~localIndexMask) == 0)
      (blockId << numLocalIndexBits) | localIndex
    }

    /** Gets the block id from an encoded index. */
    @inline
    def blockId(encoded: Int): Int = {
      encoded >>> numLocalIndexBits
    }

    /** Gets the local index from an encoded index. */
    @inline
    def localIndex(encoded: Int): Int = {
      encoded & localIndexMask
    }
  }

  /**
   * Partitioner used by ALS. We requires that getPartition is a projection. That is, for any key k,
   * we have getPartition(getPartition(k)) = getPartition(k). Since the the default HashPartitioner
   * satisfies this requirement, we simply use a type alias here.
   */
  private[recommendation] type ALSPartitioner = org.apache.spark.HashPartitioner
}
