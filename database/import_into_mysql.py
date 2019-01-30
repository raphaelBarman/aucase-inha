
import pandas as pd
import json
from sqlalchemy import create_engine
from tqdm import tqdm
tqdm.pandas()

mysql_connection_url = 'mysql://user:pass@localhost/sacase?charset=utf8'
csv_path = './sql_out'

engine = create_engine(mysql_connection_url, encoding='utf-8', echo=False)

actor_table = pd.read_csv(os.path.join(csv_path, 'actor_table.csv'))
actor_table.to_sql('actor', con=engine, if_exists='append', index=False)

sale_table = pd.read_csv(os.path.join(csv_path, 'sale_table.csv'))
sale_table.to_sql('sale', con=engine, if_exists='append', index=False)

actor_sale_table = pd.read_csv(os.path.join(csv_path, 'actor_sale_table.csv'))
table_actor_sale.rename(columns={'doc_id' : 'sale_id'}, inplace=True)
table_actor_sale.to_sql('actor_sale', con=engine, if_exists='append', index=False)

section_table = pd.read_csv(os.path.join(csv_path, 'section_table.csv'))section_table['bbox'] = section_table['bbox'].apply(str)
section_table.rename(columns = {'num_entity':'entity'}, inplace=True)
section_table.to_sql('section', con=engine, if_exists='append', index=False, chunksize=10000)

object_table = pd.read_csv(os.path.join(csv_path, 'object_table.csv'))
object_table.rename(columns = {'iif_url':'iiif_url'}, inplace=True)
object_table.to_sql('object', con=engine, if_exists='append', index=False, chunksize=10000)
