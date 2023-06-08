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

from tqdm import tqdm

def main(spark, userID):
    # Load data into DataFrame
    interactions_train = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train.parquet')
    interactions_test = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_test.parquet')
     
    # Give the dataframe a temporary view so we can run SQL queries
    interactions_train.createOrReplaceTempView('interactions_train')
    interactions_test.createOrReplaceTempView('interactions_test')
    
    interactions_train = spark.sql("SELECT DISTINCT * FROM interactions_train")
    
    interactions_total = spark.sql("SELECT DISTINCT recording_msid FROM interactions_train UNION SELECT DISTINCT recording_msid FROM interactions_test")
    interactions_total.createOrReplaceTempView('interactions_total')
    interactions_total.show()
    
    recording_map = spark.sql("""SELECT DISTINCT * FROM (SELECT recording_msid, 
                  DENSE_RANK() OVER (ORDER BY recording_msid) - 1 AS recording_intid
                  FROM interactions_total) t
                  """)
    recording_map.createOrReplaceTempView('recording_map')
    recording_map.show()
    
    
    
    # Given each user, split the data 70:30 according to timestamp 
    train_table = spark.sql("""SELECT user_id, recording_msid,  timestamp, MAX(timestamp) OVER(PARTITION BY user_id) AS time_split
                              FROM ( 
                                SELECT user_id, recording_msid, timestamp,  
                                       ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) as row_num, 
                                       COUNT(*) OVER (PARTITION BY user_id) as total_recordings, 
                                       0.7 * COUNT(*) OVER (PARTITION BY user_id) as threshold
                                       FROM interactions_train 
                                       ) t 
                              WHERE row_num <= threshold 
                              ORDER BY user_id, timestamp;""")
    
    validation_table = spark.sql("""SELECT user_id, recording_msid, timestamp, MIN(timestamp) OVER(PARTITION BY user_id) AS time_split
                                   FROM ( 
                                     SELECT user_id, recording_msid, timestamp, 
                                            ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY timestamp) as row_num, 
                                            COUNT(*) OVER (PARTITION BY user_id) as total_recordings, 
                                            0.7 * COUNT(*) OVER (PARTITION BY user_id) as threshold 
                                            FROM interactions_train 
                                            ) t 
                                   WHERE row_num > threshold 
                                   ORDER BY user_id, timestamp;""")
    # write tables into parquet files
    train_table.write.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/train.parquet')
    validation_table.write.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/val.parquet')
    interactions_test.write.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/test.parquet')
    recording_map.write.parquet(f'hdfs:/user/td2418_nyu_edu/1004-project-2023/recording_map.parquet')
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
