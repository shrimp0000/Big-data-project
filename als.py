import os

# And pyspark.sql to get the spark session
from pyspark.sql import SparkSession
from pyspark.sql.functions import count
from pyspark.sql.functions import lit
from pyspark.sql.functions import collect_list

from pyspark.sql.functions import col, max, count, row_number
from pyspark.sql.window import Window
from pyspark.sql.functions import rank, count
from pyspark.context import SparkContext
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.ml.evaluation import RegressionEvaluator, RankingEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from tqdm import tqdm
from pyspark.ml.feature import StringIndexer

def main(spark, userID):
    train_table = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/train.parquet')
    validation_table = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/val.parquet')
    test_table = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/test.parquet')
    recording_map = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/recording_map.parquet')

    
    train_table.createOrReplaceTempView('train_table')
    validation_table.createOrReplaceTempView('validation_table')
    test_table.createOrReplaceTempView('test_table')
    recording_map.createOrReplaceTempView('recording_map')

    
    train_table = spark.sql("""SELECT train_table.*, recording_map.recording_intid  
                                        FROM train_table 
                                        LEFT JOIN recording_map 
                                        ON train_table.recording_msid = recording_map.recording_msid""")
    train_table.createOrReplaceTempView('train_table')
    
    validation_table = spark.sql("""SELECT validation_table.*, recording_map.recording_intid
                                        FROM validation_table 
                                        LEFT JOIN recording_map 
                                        ON validation_table.recording_msid = recording_map.recording_msid""")
    validation_table.createOrReplaceTempView('validation_table')
    
    test_table = spark.sql("""SELECT test_table.*, recording_map.recording_intid  
                                        FROM test_table 
                                        LEFT JOIN recording_map 
                                        ON test_table.recording_msid = recording_map.recording_msid""")
    test_table.createOrReplaceTempView('test_table')
    
    # 1. dataframe transform
    train_table = spark.sql("""SELECT user_id, recording_intid, COUNT(1) AS listen_count FROM train_table GROUP BY user_id, recording_intid""")
    train_table.createOrReplaceTempView("train_table")
    
    validation_table = spark.sql("""SELECT user_id, recording_intid, COUNT(1) AS listen_count FROM validation_table GROUP BY user_id, recording_intid""")
    validation_table.createOrReplaceTempView("validation_table")
    
    test_table = spark.sql("""SELECT user_id, recording_intid, COUNT(1) AS listen_count FROM test_table GROUP BY user_id, recording_intid""")
    test_table.createOrReplaceTempView("test_table")
    
    
    
    # 2. als
    for rank in [5, 50, 500]:
        for regParam in [0.001, 1, 1000]:
            for alpha in [0.001, 1, 1000]:
                print("ALS start!!!!!!!!")
                print("rank:{}, regParam:{}, alpha:{}".format(rank,regParam, alpha))
                als = ALS(rank=rank, regParam=regParam, alpha=alpha, userCol="user_id", itemCol="recording_intid", ratingCol="listen_count", coldStartStrategy="drop", implicitPrefs=True)
                model = als.fit(train_table)

            #     param_grid = ParamGridBuilder()\
            #                  .addGrid(als.rank, [2, 4])\
            #                  .build()
            # #                  .addGrid(als.regParam, [0.01, 0.1])\
            # #                  .addGrid(als.alpha, [0.01, 0.1])\
            # #                  .build()

            #     evaluator = RegressionEvaluator(metricName="rmse", labelCol="listen_count", predictionCol="prediction")
            #     cv = CrossValidator(estimator=als, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=5)
            #     model = cv.fit(train_table)

                # 3. evaluation 

                valid_pred = model.transform(validation_table)
                valid_pred.createOrReplaceTempView("valid_pred")
                valid_pred = spark.sql("""
                            SELECT user_id, recording_intid
                            FROM (SELECT user_id, recording_intid, RANK(listen_count) OVER(PARTITION BY user_id ORDER BY listen_count DESC) AS recording_rank FROM valid_pred) t_rank
                            WHERE recording_rank <= 100 
                            ORDER BY recording_rank
                                       """)

                valid_pred = valid_pred.groupBy("user_id").agg(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
                valid_pred.createOrReplaceTempView("valid_pred")

                valid_true = spark.sql("""
                            SELECT user_id, recording_intid
                            FROM (SELECT user_id, recording_intid, RANK(listen_count) OVER(PARTITION BY user_id ORDER BY listen_count DESC) AS recording_rank FROM validation_table) t_rank
                            WHERE recording_rank <= 100 
                            ORDER BY recording_rank
                                       """)
                valid_true = valid_true.groupBy("user_id").agg(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
                valid_true.createOrReplaceTempView("valid_true")   

                val_result = spark.sql("""
                                   SELECT valid_pred.recording_intid AS pred_music, valid_true.recording_intid AS true_music
                                   FROM valid_pred JOIN valid_true 
                                   ON valid_pred.user_id = valid_true.user_id
                                   """)
                metrics = ['meanAveragePrecision', 'ndcgAtK']
                for metric in metrics:
                    eva = RankingEvaluator(predictionCol='pred_music', labelCol='true_music', metricName=metric)
                    print("Val {}:".format(metric), eva.evaluate(val_result))
        
    # 4. Test
    als = ALS(rank=5, regParam=0.001, alpha=1, userCol="user_id", itemCol="recording_intid",                ratingCol="listen_count", coldStartStrategy="drop", implicitPrefs=True)
    model = als.fit(train_table)
    test_pred = model.transform(test_table)
    test_pred.createOrReplaceTempView("test_pred")
    test_pred = spark.sql("""
                SELECT user_id, recording_intid
                FROM (SELECT user_id, recording_intid, RANK(listen_count) OVER(PARTITION BY user_id ORDER BY listen_count DESC) AS recording_rank FROM test_pred) t_rank
                WHERE recording_rank <= 100 
                           """)
    
    test_pred = test_pred.groupBy("user_id").agg(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
    test_pred.createOrReplaceTempView("test_pred")
    
    test_true = spark.sql("""
                SELECT user_id, recording_intid
                FROM (SELECT user_id, recording_intid, RANK(listen_count) OVER(PARTITION BY user_id ORDER BY listen_count DESC) AS recording_rank FROM test_table) t_rank
                WHERE recording_rank <= 100 
                           """)
    test_true = test_true.groupBy("user_id").agg(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
    test_true.createOrReplaceTempView("test_true")   

    test_result = spark.sql("""
                       SELECT test_pred.recording_intid AS pred_music, test_true.recording_intid AS true_music
                       FROM test_pred JOIN test_true 
                       ON test_pred.user_id = test_true.user_id
                       """)
    metrics = ['meanAveragePrecision', 'ndcgAtK']
    for metric in metrics:
        eva = RankingEvaluator(predictionCol='pred_music', labelCol='true_music', metricName=metric)
        print("Test {}:".format(metric), eva.evaluate(test_result))

    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
