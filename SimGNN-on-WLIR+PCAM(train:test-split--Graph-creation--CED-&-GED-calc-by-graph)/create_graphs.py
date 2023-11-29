import psycopg2
import pickle

try:
    # set-up a postgres connection
    conn = psycopg2.connect(database='ng', user='ngadmin',password='stonebra',
                                host='146.148.89.5', port=5432)
    dbcur = conn.cursor()
    print("connection successful")

    # Works
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
            )

        SELECT
            vrid, frm, hc, ubc, bbc, vid 
        FROM
            video_record
        """)
    #GROUP BY vr.video_record_id, vr.link, vr.time, lr.location, lr.address;
    #UNION

    dbcur.execute(sql)
    data = dbcur.fetchall()

    # color = dict()
    color = list()
    for row in data:
        temp = row[2]
        if temp.lower() not in color:
            color.append(temp.lower())
        #else:
        #    color[temp.lower()] += 1

        temp = row[3]
        if temp.lower() not in color:
            color.append(temp.lower())
        #else:
        #    color[temp.lower()] += 1

        temp = row[4]
        if temp.lower() not in color:
            color.append(temp.lower())
        #else:
        #    color[temp.lower()] += 1
    print(color)
    
    # Works
    import networkx as nx
    graphs = dict()
    gid = 0

    for row in data:
        node_id = 0 
        G = nx.DiGraph()
        G.add_node(0, label=0)
        node_id+=1

        G.add_node(node_id, label=1) # type - name of Entity
        G.add_edge(0, node_id, label=0) # label=0 is hasEntity
        node_id+=1
        G.add_node(node_id, label=2) # type - entityType
        G.add_edge(node_id-1, node_id, label=1) # entityType
        node_id+=1

        G.add_node(node_id, label=3+color.index(row[2].lower())) # type - color
        G.add_edge(1, node_id, label=2) # head_color
        node_id+=1
        G.add_node(node_id, label=3+color.index(row[3].lower()))
        G.add_edge(1, node_id, label=3) # ubc
        node_id+=1
        G.add_node(node_id, label=3+color.index(row[4].lower()))
        G.add_edge(1, node_id, label=4) # lbc
        node_id+=1

        G.graph['gid'] = gid
        graphs[row[5]] = G
        gid += 1
        
    # Works
    sql = dbcur.mogrify("""
        set seed to 0.1234; /* seed (-1 .. 1) */
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
            irrid, hc, ubc, bbc, g, iid 
        FROM
            incident_record
        """)


    dbcur.execute(sql)
    data = dbcur.fetchall()

    for row in data:
        for temp in row[1]:
            if temp.lower() not in color:
                color.append(temp.lower())
            #else:
            #    color[temp.lower()] += 1

        for temp in row[2]:
            if temp.lower() not in color:
                color.append(temp.lower())
            #else:
            #    color[temp.lower()] += 1

        for temp in row[3]:
            if temp.lower() not in color:
                color.append(temp.lower())
            #else:
            #    color[temp.lower()] += 1
    print(color)
    
    gender =['male', 'female']
    
    #works
    import networkx as nx

    for row in data:
        # print(row)
        G = nx.DiGraph()
        node_id = 0  
        # if-else; if frame is not None
        G.add_node(0, label=0)
        node_id+=1
        # if-else done
        G.add_node(1, label=1) # type - name of Entity
        G.add_edge(0, node_id, label=0) # label=0 is hasEntity
        node_id+=1
        G.add_node(node_id, label=2) # type - entityType
        G.add_edge(node_id-1, node_id, label=1) # entityType
        node_id+=1
        ###
        for item in row[1]:
            G.add_node(node_id, label=3+color.index(item.lower())) # type - color
            G.add_edge(1, node_id, label=2) # head_color
            node_id+=1
        for item in row[2]:
            G.add_node(node_id, label=3+color.index(item.lower()))
            G.add_edge(1, node_id, label=3) # ubc
            node_id+=1
        for item in row[3]:
            G.add_node(node_id, label=3+color.index(item.lower()))
            G.add_edge(1, node_id, label=4) # lbc
            node_id+=1
        G.add_node(node_id, label=3+len(color)+gender.index(row[4].lower()))
        G.add_edge(1, node_id, label=5) # gender
        node_id+=1

        #print(G.nodes(data=True))
        #print(G.edges(data=True))

        G.graph['gid'] = gid
        graphs[row[5]] = G
        gid += 1
        
    with open("data/graphs_v2.pkl", "wb") as f:
        pickle.dump(graphs, f)

except (Exception, psycopg2.Error) as error :
    if(conn):
        print("Failed to insert record into mobile table", error)

finally:
    if(conn):
        dbcur.close()
        conn.close()

