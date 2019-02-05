import pandas as pd
import os
from sqlalchemy import create_engine
from tqdm import tqdm
tqdm.pandas()


def import_into_mysql(config):
    mysql_host = config['database']['mysql_host']
    mysql_schema = config['database']['mysql_schema']
    mysql_user = config['database']['mysql_user']
    mysql_pass = config['database']['mysql_pass']
    csv_path = config['database']['sql_output_dir']

    mysql_connection_url = 'mysql://%s:%s@%s/%s?charset=utf8' % (
        mysql_user, mysql_pass, mysql_host, mysql_schema)

    engine = create_engine(mysql_connection_url, encoding='utf-8', echo=False)

    actor_table = pd.read_csv(os.path.join(csv_path, 'actor_table.csv'))
    actor_table.to_sql('actor', con=engine, if_exists='append', index=False)

    sale_table = pd.read_csv(os.path.join(csv_path, 'sale_table.csv'))
    sale_table.to_sql('sale', con=engine, if_exists='append', index=False)

    actor_sale_table = pd.read_csv(
        os.path.join(csv_path, 'actor_sale_table.csv'))
    actor_sale_table.rename(columns={'doc_id': 'sale_id'}, inplace=True)
    actor_sale_table.to_sql(
        'actor_sale', con=engine, if_exists='append', index=False)

    section_table = pd.read_csv(os.path.join(csv_path, 'section_table.csv'))
    section_table['bbox'] = section_table['bbox'].apply(str)
    section_table.rename(columns={'num_entity': 'entity'}, inplace=True)
    section_table.to_sql(
        'section',
        con=engine,
        if_exists='append',
        index=False,
        chunksize=10000)

    object_table = pd.read_csv(os.path.join(csv_path, 'object_table.csv'))
    object_table.rename(columns={'iif_url': 'iiif_url'}, inplace=True)
    object_table.to_sql(
        'object', con=engine, if_exists='append', index=False, chunksize=10000)
