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

from tqdm import tqdm

def main(spark, userID):
    interactions_train_small = spark.read.parquet(f'hdfs:/user/bm106_nyu_edu/1004-project-2023/interactions_train_small.parquet')
    interactions_train_small.createOrReplaceTempView('interactions_train_small')
#     spark.sql("""SELECT COUNT(*) FROM interactions_train_small WHERE user_id IS NULL OR recording_msid IS NULL OR timestamp IS NULL;""").show()
#     spark.sql("""SELECT user_id, recording_msid, timestamp, COUNT(*) as duplicate_count
#                  FROM interactions_train_small
#                  GROUP BY user_id, recording_msid, timestamp
#                  HAVING COUNT(*) > 1;""").show()
    duplicates = spark.sql("""SELECT COUNT(*)-1 as duplicate_count
                     FROM interactions_train_small
                     GROUP BY user_id, recording_msid, timestamp
                     HAVING COUNT(*) > 1""")
    duplicates.createOrReplaceTempView('duplicates')
    spark.sql("""SELECT SUM(duplicate_count) as total_duplicates
                 FROM duplicates;""").show()

if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)