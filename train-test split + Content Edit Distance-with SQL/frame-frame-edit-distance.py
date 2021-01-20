import psycopg2

try:
    # set-up a postgres connection
    conn = psycopg2.connect(database='ng', user='ngadmin',password='stonebra',
                                host='146.148.89.5', port=5432)
    dbcur = conn.cursor()
    print("connection successful")

    sql = dbcur.mogrify("""
    set seed to 0.1234; /* seed (-1 .. 1) */
    WITH 
        video_record AS(
            select 
                dp.vid,
                vr.video_record_id as vrid, vr.time as time, dp.frame_num as frm, 
                dp.head_color as hc, 
                dp.upper_body_color as ubc, 
                dp.bottom_body_color as bbc,
                dp.gender
            from 
                video_record as vr
            JOIN 
                detected_people as dp
            ON 
                vr.video_record_id = dp.video_record_id
            JOIN 
                location_record as lr
            ON 
                vr.video_record_id = lr.video_record_id
            AND 
                vr.time >= '2019-11-14'::date and vr.time <= '2019-12-16'::date 
            ORDER BY RANDOM()
            LIMIT 1450
            /*
            ORDER BY 
                vr.video_record_id, dp.frame_num
            LIMIT 20000
            */
        )

    INSERT INTO content_edit_distance_realistic (ced_type, uid1, uid2, distance)
    (
        SELECT 
            0 as ced_type, A.vid as uid1, B.vid as uid2,
            (CASE WHEN A.ubc=B.ubc THEN 0
                        ELSE 1
                   END) + 
            (CASE WHEN A.hc=B.hc THEN 0
                        ELSE 1
                   END) + 
            (CASE WHEN A.bbc=B.bbc THEN 0
                        ELSE 1
                   END)
            AS distance
        FROM video_record A
        JOIN video_record B 
        ON 
            A.vid != B.vid
            AND 
            A.frm != B.frm
    )
    ON CONFLICT (ced_type, uid1, uid2)
    DO NOTHING
    ;""")

    dbcur.execute(sql)
    #data = dbcur.fetchall()
    leng = 0
    # print(len(data))

    conn.commit()
    count = dbcur.rowcount
    print (count, " Record inserted successfully into mobile table")

except (Exception, psycopg2.Error) as error :
    if(conn):
        print("Failed to insert record into mobile table", error)

finally:
    if(conn):
        dbcur.close()
        conn.close()

