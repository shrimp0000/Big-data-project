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
import numpy as np
import scipy.sparse as sp
from pyspark.sql import functions as F

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
    
#     interactions_total = spark.sql("SELECT DISTINCT user_id, recording_intid FROM train_table UNION SELECT DISTINCT user_id, recording_intid FROM validation_table UNION SELECT DISTINCT user_id, recording_intid FROM test_table")
#     interactions_total.createOrReplaceTempView('interactions_total')
#     interactions_total.show()
    
    pandas_df = train_table.toPandas()
    
    user_ids = pandas_df['user_id'].unique()
    item_ids = pandas_df['recording_intid'].unique()
    
    user_id_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    
    # create train sparse matrix
    interaction_sparse_train = sp.lil_matrix((len(user_ids), len(item_ids)), dtype=np.int32)
    for row in train_table.itertuples(index=False):
        interaction_sparse_train[user_id_to_index[row.user_id], row.recording_intid] = row.listen_count
        
    model = LightFM(loss='warp')
    model.fit(interaction_sparse_train)
    train_precision = precision_at_k(model, interaction_sparse_train, k=100).mean()
    print("train mAP: ", train_precision)
    
# Only enter this block if we're in main
if __name__ == "__main__":

    # Create the spark session object
    spark = SparkSession.builder.appName('part1').getOrCreate()

    # Get user userID from the command line
    # We need this to access the user's folder in HDFS
    userID = os.environ['USER']

    # Call our main routine
    main(spark, userID)
