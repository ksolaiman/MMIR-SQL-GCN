import psycopg2

try:
    # set-up a postgres connection
    conn = psycopg2.connect(database='ng', user='ngadmin',password='stonebra',
                                host='146.148.89.5', port=5432)
    dbcur = conn.cursor()
    print("connection successful")

    sql = dbcur.mogrify("""
    WITH 
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

    SELECT
        A.iid, B.iid, A.g, B.g, lower(A.ubc::text)::text[], lower(B.ubc::text)::text[], 
        lower(A.hc::text)::text[], lower(B.hc::text)::text[], 
        lower(A.bbc::text)::text[], lower(B.bbc::text)::text[]
    FROM incident_record A
    JOIN incident_record B 
    ON
    A.irrid != B.irrid
    ;
    """)

    dbcur.execute(sql)
    data = dbcur.fetchall()
    leng = 0
    result = []
    leng = 0
    for row in data:
        #leng+=1
        result.append(
            (
              2, row[0], row[1], 
              editDistDP(row[4], row[5], len(row[4]), len(row[5])) +
              editDistDP(row[6], row[7], len(row[6]), len(row[7])) +
              editDistDP(row[8], row[9], len(row[8]), len(row[9])) + 
              (1 if row[2] != row[3] else 0)
             )
        )
    print(len(result))
    
    for item in result:
        sql = dbcur.mogrify('INSERT INTO content_edit_distance_realistic (ced_type, uid1, uid2, distance) VALUES (%s, %s, %s, %s) ON CONFLICT DO NOTHING RETURNING ced_id', item)
        dbcur.execute(sql)

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

