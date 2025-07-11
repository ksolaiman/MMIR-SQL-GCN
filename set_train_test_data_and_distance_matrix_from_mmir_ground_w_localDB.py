import psycopg2
import numpy as np
import pickle

def calc_distance_local_DB(testset):
    '''
    testset: a list containing the id of the mmir obj. in mmir_ground table
    '''
    item2idx = dict()
    idx2item = dict()
    for idx, item in enumerate(testset):
        item2idx[item]=idx
        idx2item[idx]=item
        
    dist = np.zeros((len(testset), len(testset)))
    print(dist.shape)
    
    for i in range(0, len(testset)):
        for j in range(0, len(testset)):
            if i == j:
                dist[i][j] = np.inf

    # set-up a postgres connection
    conn = psycopg2.connect(database='Salvi', user='Salvi', password='sholock',
                            host='localhost', port=5433)
    dbcur = conn.cursor()
    print("connection successful")
    sql = dbcur.mogrify("""
    SELECT 
            A.mgid as Aid, B.mgid as Bid,
            CONCAT(A.ubc::character varying, A.lbc::character varying, 
	   A.gender::character varying) as Alabel,
	   CONCAT(B.ubc::character varying, B.lbc::character varying, 
	   B.gender::character varying) as Blabel,
	   (CASE WHEN A.ubc=B.ubc THEN 0
                        ELSE 1
                   END) + 
            (CASE WHEN A.gender=B.gender THEN 0
                        ELSE 3
                   END) + 
            (CASE WHEN A.lbc=B.lbc THEN 0
                        ELSE 2
                   END)
            AS distance
        FROM mmir_ground A
        JOIN mmir_ground B 
        ON 
            A.mgid != B.mgid
			and 
			A.ubc!= -1 and A.lbc!= -1 and A.ubc!= 100 and A.lbc!= 100
			and 
			B.ubc!= -1 and B.lbc!= -1 and B.ubc!= 100 and B.lbc!= 100
			and A.mgid in %s --(6, 2)
			and B.mgid in %s --(5, 3)
--order by A.mgid, distance asc

--Limit 1000
    """, (tuple(testset), tuple(testset))) #(tuple(testset[0:3]), tuple(testset[2:5])))
    dbcur.execute(sql)
    rows = dbcur.fetchall()
    
    for item in rows:
        print(item)
        dist[item2idx[item[0]]][item2idx[item[1]]] = item[4]
    
    return dist, item2idx, idx2item

def prepare_dataset():
    # set-up a postgres connection
    conn = psycopg2.connect(database='Salvi', user='Salvi',password='sholock',
                                host='localhost', port=5433)
    dbcur = conn.cursor()
    print("connection successful")
    try:
        sql = dbcur.mogrify("""
        set seed to 0.1234;
        select --unnest(itest), unnest(itrain) --, 
        --itrain || vtrain || ttrain as trainsetByClass, array_length(itrain || vtrain || ttrain, 1), label
        unnest(itrain || vtrain || ttrain) as trainset,
        unnest(ivalid || vvalid || tvalid) as validset,
        unnest(itest || vtest || ttest) as testset,
        unnest(itest) as imgtestset,
        unnest(vtest) as videotestset,
        unnest(ttest) as texttestset,
        unnest(itrain) as imgtrainset,
        unnest(vtrain) as videotrainset,
        unnest(ttrain) as texttrainset,
        unnest(ivalid) as imgvalidset,
        unnest(vvalid) as videovalidset,
        unnest(tvalid) as textvalidset
    --ivalid || vvalid || tvalid as validset
    --sum(array_length(vtest, 1)) + sum(array_length(itest, 1)) + sum(array_length(ttest, 1)) as test_size
    --, 
    --sum(itrainlen), sum(ivalidlen)
    -- each one of the upper_command is right, just gives different results
    from
    (select *,
    array_length(imageids, 1),
    array_length(videoids, 1),
    imageids[0: floor(array_length(imageids, 1)*0.6)] as itrain,
    array_length(imageids[0: floor(array_length(imageids, 1)*0.6)], 1) as itrainlen,
    imageids[floor(array_length(imageids, 1)*0.6)+1:floor(array_length(imageids, 1)*0.8)] as ivalid,
    array_length(imageids[floor(array_length(imageids, 1)*0.6)+1:floor(array_length(imageids, 1)*0.8)], 1) as ivalidlen,
    imageids[floor(array_length(imageids, 1)*0.8)+1:] as itest,
    array_length(imageids[floor(array_length(imageids, 1)*0.8)+1:], 1) as itestlen,

    textids[0: floor(array_length(textids, 1)*0.6)] as ttrain,
    textids[floor(array_length(textids, 1)*0.6)+1:floor(array_length(videoids, 1)*0.8)] as tvalid,
    textids[floor(array_length(textids, 1)*0.8)+1:] as ttest,

    videoids[0: floor(array_length(videoids, 1)*0.6)] as vtrain,
    videoids[floor(array_length(videoids, 1)*0.6)+1:floor(array_length(videoids, 1)*0.8)] as vvalid,
    videoids[floor(array_length(videoids, 1)*0.8)+1:] as vtest
    from (
        select 
        string_agg(mgid::varchar, ', ') as ids, 
        -- string_agg(ctype::character varying, ', ') as ctypes,
        COUNT(CASE WHEN ctype='image' THEN 1 END) as nimage,
        COUNT(CASE WHEN ctype='video' THEN 1 END) as nvideo,
        COUNT(CASE WHEN ctype='untext' THEN 1 END) as ntext,
        array_remove(array_agg(CASE WHEN ctype='image' 
                               THEN mgid END 
                               order by random()), NULL) as imageids,
        array_remove(array_agg(CASE WHEN ctype='image' 
                               THEN mgid END), NULL) as imageids2,
        array_remove(array_agg(CASE WHEN ctype='video' 
                               THEN mgid END
                               order by random()), NULL) as videoids,
        array_remove(array_agg(CASE WHEN ctype='untext' 
                               THEN mgid END
                               order by random()), NULL) as textids,
        ubc, lbc, gender, 
        count(*) as freq,
        CONCAT(ubc::character varying, lbc::character varying, 
               gender::character varying) as label
        from mmir_ground
        where ubc!= -1 and lbc!= -1 and ubc!= 100 and lbc!= 100 
        -- for protection against text attributes
        --and ctype='untext'
        group by ubc, lbc, gender
        order by freq desc
    )fullTable
    where freq >= 10
    -- and nimage != 0 and nvideo != 0 and ntext != 0
     ) idstable
        """)
        dbcur.execute(sql)
        rows = dbcur.fetchall()
    
    except (Exception, psycopg2.Error) as error :
        print ("Error while fetching data from PostgreSQL", error)

    finally:
        #closing database connection.
        if(conn):
            dbcur.close()
            conn.close()
            print("PostgreSQL connection is closed")

    return rows

def write_dataset(testset, item2idx, idx2item, name):
        with open(name+".pkl", "wb") as f:
            pickle.dump(testset, f)
        with open(name+"item2idx.pkl", "wb") as f:
            pickle.dump(item2idx, f)
        with open(name+"idx2item.pkl", "wb") as f:
            pickle.dump(idx2item, f)
    
def main():
    rows = prepare_dataset()
    trainset = [row[0] for row in rows]
    print(len(trainset))
    testset = [row[2] for row in rows if row[2]]
    validset = [row[1] for row in rows if row[1]] # coz train is longer in length, tet column has null/none values
    print(len(validset))
    print(len(testset))
    print(testset[1827])

    #np.random.seed(42)
    #np.random.shuffle(testset) # need to shuffle before testing as all same class labels are concatanated one after another.

    from datetime import datetime

    start = datetime.now()
    dist, item2idx, idx2item = calc_distance_local_DB(testset)
    end = datetime.now()
    print(end -  start)

    '''
    write_dataset(testset, item2idx, idx2item, 'testset')
    with open("test_distance_matrix.pkl", "wb") as f:
            pickle.dump(dist, f)

    dist, item2idx, idx2item = calc_distance(validset)
    write_dataset(validset, item2idx, idx2item, 'validset')

    with open("valid_distance_matrix.pkl", "wb") as f:
            pickle.dump(dist, f)

    dist, item2idx, idx2item = calc_distance(trainset)
    write_dataset(trainset, item2idx, idx2item, 'trainset')
    with open("train_distance_matrix.pkl", "wb") as f:
            pickle.dump(dist, f)
    '''

if __name__ == "__main__":
    main()