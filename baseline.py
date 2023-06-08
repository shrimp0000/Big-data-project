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


from tqdm import tqdm

def main(spark, userID):
    # Load data into DataFrame
    original_train_table = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/train.parquet')
    validation_table = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/val.parquet')
    test_table = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/test.parquet')
    recording_map = spark.read.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/recording_map.parquet')
    
    original_train_table.createOrReplaceTempView('original_train_table')
    validation_table.createOrReplaceTempView('validation_table')
    test_table.createOrReplaceTempView('test_table')
    recording_map.createOrReplaceTempView('recording_map')
    
    original_train_table = spark.sql("""SELECT original_train_table.*, recording_map.recording_intid  
                                        FROM original_train_table 
                                        LEFT JOIN recording_map 
                                        ON original_train_table.recording_msid = recording_map.recording_msid""")
    original_train_table.createOrReplaceTempView('original_train_table')
    
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

    train_table = spark.sql(""" SELECT DISTINCT user_id, recording_intid FROM original_train_table""")
    train_grouped = train_table.groupBy("user_id").agg(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
    train_grouped.createOrReplaceTempView('train_grouped')
    
    validation_table = spark.sql(""" SELECT DISTINCT user_id, recording_intid FROM validation_table""")
    val_grouped = validation_table.groupBy("user_id").agg(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
    val_grouped.createOrReplaceTempView('val_grouped')
    
    test_table = spark.sql(""" SELECT DISTINCT user_id, recording_intid FROM test_table""")
    test_grouped = test_table.groupBy("user_id").agg(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
    test_grouped.createOrReplaceTempView('test_grouped')
    
    
    betas = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    for beta in betas:
        print("Beta:", beta)
        top100_table = spark.sql("""SELECT recording_intid FROM original_train_table GROUP BY recording_intid ORDER BY COUNT(1)/(COUNT(DISTINCT user_id) + {}) DESC LIMIT 100""".format(beta))

        top100_table = top100_table.select(collect_list(col("recording_intid").cast("double")).alias("recording_intid"))
        top100_table.createOrReplaceTempView('top100_table')

        train_result = spark.sql("""
                       SELECT train_grouped.recording_intid as true_music, top100_table.recording_intid AS pred_music
                       FROM train_grouped CROSS JOIN top100_table
                       """)
        metrics = ['meanAveragePrecision', 'ndcgAtK']
        for metric in metrics:
            eva = RankingEvaluator(predictionCol='pred_music', labelCol='true_music', metricName=metric)
            print("Train {}:".format(metric), eva.evaluate(train_result))
            
        val_result = spark.sql("""
                       SELECT val_grouped.recording_intid as true_music, top100_table.recording_intid AS pred_music
                       FROM val_grouped CROSS JOIN top100_table
                       """)
        metrics = ['meanAveragePrecision', 'ndcgAtK']
        for metric in metrics:
            eva = RankingEvaluator(predictionCol='pred_music', labelCol='true_music', metricName=metric)
            print("Val {}:".format(metric), eva.evaluate(val_result))
            
        test_result = spark.sql("""
                       SELECT test_grouped.recording_intid as true_music, top100_table.recording_intid AS pred_music
                       FROM test_grouped CROSS JOIN top100_table
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
