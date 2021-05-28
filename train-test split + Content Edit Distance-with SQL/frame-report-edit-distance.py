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
            LIMIT 20 
            */
        ),
        incident_record AS(
            select 
                irr.incident_report_id as iid,
                irr.incident_report_id_v4 as irrid, 
                irr.incident_date + irr.incident_time,
                null, 
                detected_people.head_color as hc,
                detected_people.upper_body_color as ubc,
                detected_people.bottom_body_color as bbc,
                detected_people.gender as g
            from 
                incident_report_record as irr
            JOIN 
                incident_report_detected_people as detected_people
            ON 
                irr.incident_report_id = detected_people.incident_report_id
            /* 
                Following erases the rows which have none of the color codes
                That kind of report will match with all frames, so no need
            */
            AND
                (
                    array_length(detected_people.head_color, 1) is not null OR
                    array_length(detected_people.upper_body_color, 1) is not null OR
                    array_length(detected_people.bottom_body_color, 1) is not null
                )
        )

    INSERT INTO content_edit_distance_realistic (ced_type, uid1, uid2, distance)
    (
        SELECT 
            1 as ced_type, B.vid as uid1, A.iid as uid2, 
            /* no need of next line as filtered out the rows with all null color */
            /* CASE WHEN (array_length(A.bbc, 1) is null and array_length(A.ubc, 1) is null and array_length(A.hc, 1) is null) THEN 100
                ELSE
            */
            (CASE WHEN lower(B.bbc) ILIKE ANY(A.bbc) THEN 0
                    WHEN (array_length(A.bbc, 1) is null and B.bbc is null) THEN 0
                    WHEN array_length(A.bbc, 1) is null THEN 1
                        ELSE array_length(A.bbc, 1)
                   END) +
            (CASE WHEN lower(B.ubc) ILIKE ANY(A.ubc) THEN 0
                    WHEN (array_length(A.ubc, 1) is null and B.ubc is null) THEN 0
                    WHEN array_length(A.bbc, 1) is null THEN 1
                        ELSE array_length(A.bbc, 1)
                   END) +
            (CASE WHEN lower(B.hc) ILIKE ANY(A.hc) THEN 0
                    WHEN (array_length(A.hc, 1) is null and B.hc is null) THEN 0
                    WHEN array_length(A.bbc, 1) is null THEN 1
                        ELSE array_length(A.bbc, 1)
                   END) + 
            /* video records dont have gender, so always +1 to add it */
            1 
            /* END */
            AS distance
        FROM incident_record A
        JOIN video_record B
        ON
            A.irrid != B.vrid
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

