import java.io.{File, PrintWriter}
import java.security.MessageDigest

import org.apache.spark.broadcast.Broadcast
import org.apache.spark.mllib.recommendation.ALS
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{Row, SQLContext}
import org.apache.spark.{SparkConf, SparkContext}

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.Random
import scala.util.hashing.{Hashing, MurmurHash3}

/**
  * Created by lambdachen on 16/7/2.
  */
object RunRecommender {

  val SEP = "#"
  var loggerFile: PrintWriter = null

  def main(args: Array[String]): Unit ={
    val conf = new SparkConf().setAppName("recommend heros")
    val sc = new SparkContext(conf)
    val sqlContext = new SQLContext(sc)

    loggerFile = new PrintWriter(new File("run_recommender.log"))
//    val infoDataDir = "/Users/lambdachen/data/game/wangzherongyao/wangzhe_tbHeroInfo_10.217.176.48"
    val infoDataDir = "E:\\tmp\\wangzhe_tbHeroInfo_10.217.176.48"
    recommender(sc, sqlContext, infoDataDir)
    loggerFile.close()
  }

  private def readHeroInfo(sc: SparkContext,
                           sqlContext: SQLContext,
                           infoDataDir: String): Unit ={
    val infoData = sqlContext.read.parquet(infoDataDir)
  }

  private def recommender(sc: SparkContext,
                          sqlContext: SQLContext,
                          infoDataDir: String): Unit ={
    val ratings = buildRatings1(sqlContext, infoDataDir).cache()
    val Array(trainData, cvData) = ratings.randomSplit(Array(0.8, 0.2))
    val allHeroIds = ratings.map{rat => rat.product}.distinct().collect().toSeq
    ratings.unpersist()
    loggerFile.write("after ratings.unpersist()")

    trainData.cache()
    cvData.cache()

//    val (trainData, cvData) = buildTrainCVData(sqlContext, infoDataDir)
//    val allHeroIds = (trainData.map{rat => rat.product}.collect().toSet ++
//      cvData.map{rat => rat.product}.collect().toSet).toSeq

    val bAllHeroIds = sc.broadcast(allHeroIds)

    val rank = 10            //number of features to use
    val lambda = 1.5
    val alpha = 1.0
    val iterations = 20
    val alsModel = ALS.trainImplicit(trainData, rank, iterations, lambda, alpha)
    loggerFile.write("alsModel computation end")

    val (auc, posCount, negCount, posPredNum, negPredNum) = areaUnderCurve(cvData, bAllHeroIds, alsModel.predict)
    loggerFile.write("areaUnderCurve computation end")

    val mse = computeMSE(cvData, alsModel.predict)
    loggerFile.write("computeMSE computation end")

    println(s"trainData.count: ${trainData.count()}")
    println(s"cvData.count: ${cvData.count()}")
    println(s"auc: $auc")
    println(s"posCount: $posCount")
    println(s"negCount: $negCount")
    println(s"posPredNum: $posPredNum")
    println(s"negPredNum: $negPredNum")
    println(s"mse: $mse")

    trainData.unpersist()
    cvData.unpersist()
  }

  private def buildTrainCVData(sqlContext: SQLContext,
                               infoDataDir: String): (RDD[Rating], RDD[Rating]) ={
    val infoData = sqlContext.read.parquet(infoDataDir)
    val infoTmpTable = "hero_info_data"
    infoData.registerTempTable(infoTmpTable)
    val sqlText = s"select Uid, LogicWorldID, HeroInfo.HeroNum, HeroInfo.HeroInfoList.CommonInfo.HeroID from $infoTmpTable"
    val rawRDD = sqlContext.sql(sqlText).map{
      r =>
        val strId = r(0) + SEP + r(1)
        (strId, r(3).asInstanceOf[mutable.WrappedArray[Long]])
    }
    val Array(trainRawData, cvRawData) = rawRDD.randomSplit(Array(0.8, 0.2))

    val trainUids = trainRawData.map(t => getHashedId(t._1)).collect().toSet
    val cvUids = cvRawData.map(t => getHashedId(t._1)).collect().toSet
    val interSet = trainUids.intersect(cvUids)

    def constructRatings(rawTuples: RDD[(String, mutable.WrappedArray[Long])]) = {
      rawTuples.map{
        case (strId, heroIds) =>
          val hashId = getHashedId(strId)
          if(!interSet.contains(hashId)){
            val playerRatings = heroIds.map{
              hid =>
                Rating(getHashedId(strId), hid.toInt, 1.0)
            }
            playerRatings
          }else{
            Seq.empty[Rating]
          }
      }.flatMap{case list => list.view.map(r => r)}
    }

    val trainRatings = constructRatings(trainRawData)
    val cvRatings = constructRatings(cvRawData)
    (trainRatings, cvRatings)
  }

  private def buildRatings2(sqlContext: SQLContext,
                            infoDataDir: String): RDD[Rating] = {
    val infoData = sqlContext.read.parquet(infoDataDir)
    val infoTmpTable = "hero_info_table"
    infoData.registerTempTable(infoTmpTable)

    val sqlText = s"select Uid, LogicWorldID, HeroInfo.HeroNum, HeroInfo.HeroInfoList.CommonInfo.HeroID, " +
      s"HeroInfo.HeroInfoList.CommonInfo.GameWinNum, HeroInfo.HeroInfoList.CommonInfo.GameLoseNum from $infoTmpTable"
    val ratings = sqlContext.sql(sqlText).map{
      r =>
        val strId = r(0) + SEP + r(1)
        val playerRating = r(3).asInstanceOf[mutable.WrappedArray[Long]].zip(
          r(4).asInstanceOf[mutable.WrappedArray[Long]]
        ).zip(
          r(5).asInstanceOf[mutable.WrappedArray[Long]]
        ).map{
          case ((heroId, winNum), loseNum) =>
            Rating(getHashedId(strId), heroId.toString.toInt, math.log(1.0 + winNum.toDouble + loseNum.toDouble))
        }
        playerRating
    }.flatMap({case (list) => list.view.map(r => r)})

    ratings
  }

  private def buildRatings1(sqlContext: SQLContext,
                            infoDataDir: String): RDD[Rating] = {
    val infoData = sqlContext.read.parquet(infoDataDir)
    val infoTmpTable = "hero_info_table"
    infoData.registerTempTable(infoTmpTable)
    val sqlText = s"select Uid, LogicWorldID, HeroInfo.HeroNum, HeroInfo.HeroInfoList.CommonInfo.HeroID from $infoTmpTable"
      //s"where HeroInfo.HeroNum > 5 and HeroInfo.HeroNum <= 45"

    val ratings = sqlContext.sql(sqlText).map{
      r =>
        val strId = r(0) + SEP + r(1)
        val playerRatings = r(3).asInstanceOf[mutable.WrappedArray[Long]].map{
          heroId =>
            Rating(getHashedId(strId), heroId.toString.toInt, 1.0)
        }
        playerRatings
    }.flatMap({case (list) => list.view.map(r => r)})

    ratings
  }

  private def areaUnderCurve(cvData: RDD[Rating],
                             bAllHeroIds: Broadcast[Seq[Int]],
                             predictFunction: (RDD[(Int, Int)] => RDD[Rating])) = {
    val positiveUserProducts = cvData.map(r => (r.user, r.product))
    val positivePredictions = predictFunction(positiveUserProducts).groupBy(_.user).cache()

    //construct negative heroids for player
    val negativeUserProducts = positiveUserProducts.groupByKey().mapPartitions{
      userIdAndPosHeroIDs => {
        val random = new Random()
        val allHeroIds = bAllHeroIds.value
        userIdAndPosHeroIDs.map{
          case (uid, posHeroIDs) =>
            val posHeroIDSet = posHeroIDs.toSet
            var i = 0
            val negativeHeros = new ArrayBuffer[Int]
            while(i < allHeroIds.size && negativeHeros.size < posHeroIDSet.size){
              val hid = allHeroIds(random.nextInt(allHeroIds.size))
              if(!posHeroIDSet.contains(hid)){
                negativeHeros += hid
              }
              i += 1
            }
            negativeHeros.map(hid => (uid, hid))
        }
      }
    }.flatMap(t => t)

    val negativePredictions = predictFunction(negativeUserProducts).groupBy(_.user).cache()

    //join positive and negative by user
    val meanAUC = positivePredictions.join(negativePredictions).values.map{
      case (positiveRatings, negativeRatings) =>
        var correct = 0
        var total = 0
        for(pos <- positiveRatings;
            neg <- negativeRatings){
          if(pos.rating > neg.rating)
            correct += 1
          total += 1
        }
        correct.toDouble / total.toDouble
    }.mean()

    val posUPNum = positiveUserProducts.count()
    val negUPNum = negativeUserProducts.count()
    val positivePredNum = positivePredictions.map{case (id, rs) => rs.size}.reduce(_ + _)
    val negPredNum = negativePredictions.map{case (id, rs) => rs.size}.reduce(_ + _)

//    positiveUserProducts.unpersist()
//    negativeUserProducts.unpersist()
    positivePredictions.unpersist()
    negativePredictions.unpersist()

    (meanAUC, posUPNum, negUPNum, positivePredNum, negPredNum)
  }

  private def computeMSE(positiveData: RDD[Rating],
                         predictFunction: (RDD[(Int, Int)] => RDD[Rating])) = {
    val userHeros = positiveData.map{case Rating(user, hero, rate) => (user, hero)}
    val predictions = predictFunction(userHeros).map(r => ((r.user, r.product), r.rating))
    val posData = positiveData.map(r => ((r.user, r.product), r.rating))
    val ratesAndPred = posData.join(predictions).map{
      case ((user, product), (r1, r2)) =>
        val err = (r1 - r2)
        err * err
    }.mean()
    ratesAndPred
  }

  private def getHashedId(strId: String): Int = {
    val md5 = MessageDigest.getInstance("MD5").digest(strId.getBytes).map(0xff & _).map{
      "%02x".format(_)
    }.foldLeft(""){_ + _}
    MurmurHash3.stringHash(md5)
  }

}
