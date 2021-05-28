import psycopg2

def ced_calc_frames_frames(a,b):
    ced = 0
    if a[0]!=b[0]:
        ced+=1
    if a[1]!=b[1]:
        ced+=1
    if a[2]!=b[2]:
        ced+=1
    return ced
# ced_calc_frames_frames(a,b)

try:
    # set-up a postgres connection
    conn = psycopg2.connect(database='ng', user='ngadmin',password='stonebra',
                                host='146.148.89.5', port=5432)
    dbcur = conn.cursor()
    print("connection successful")

    # create train-test-folder
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

        SELECT * from video_record
        ORDER BY RANDOM()
        """);

    dbcur.execute(sql)
    data = dbcur.fetchall()
    frames = [item[0] for item in data]
    frecs = {item[0]:(item[4], item[5], item[6])for item in data}
    train = [item[0] for item in data[0:870]]
    validation = [item[0] for item in data[870:870+290]]
    test = [item[0] for item in data[870+290:]]

    print(len(test))

    # create train-test-folder
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

        SELECT iid from incident_record
        ORDER BY RANDOM()
        """);

    dbcur.execute(sql)
    data = dbcur.fetchall()
    reports = [item[0] for item in data]
    print(len(data))
    train += [item[0] for item in data[0:27]]
    validation += [item[0] for item in data[27:27+10]]
    test += [item[0] for item in data[27+10:]]

    print(len(train))
    print(len(validation))
    print(len(test))
    
    ### WORKS
    sql = dbcur.mogrify("""
        SELECT * FROM public.content_edit_distance_realistic
        """);

    dbcur.execute(sql)
    ceds = dbcur.fetchall()
    print(len(ceds))
    
    ceds_dict = dict()
    for item in ceds:
        ceds_dict[(item[2], item[3])] = item[4]
        ceds_dict[(item[3], item[2])] = item[4]
        
    problems = list()
    train_data = list()

    for i in train:
        for j in train:
            if i != j:
                try:
                    #print(ceds_dict[(i, j)])
                    train_data.append((i, j, ceds_dict[(i, j)]))
                except:
                    if i in frames and j in frames:
                        #print(ced_calc_frames_frames(frecs[i], frecs[j]))
                        train_data.append((i, j, ced_calc_frames_frames(frecs[i], frecs[j])))
                    else:
                        problems.append((i,j))
                        
    problems = list()
    val_data = list()

    for i in validation:
        for j in validation:
            if i != j:
                try:
                    #print(ceds_dict[(i, j)])
                    val_data.append((i, j, ceds_dict[(i, j)]))
                except:
                    if i in frames and j in frames:
                        #print(ced_calc_frames_frames(frecs[i], frecs[j]))
                        val_data.append((i, j, ced_calc_frames_frames(frecs[i], frecs[j])))
                    else:
                        problems.append((i,j))
                        
    problems = list()
    test_data = list()

    for i in test:
        for j in test:
            if i != j:
                try:
                    #print(ceds_dict[(i, j)])
                    test_data.append((i, j, ceds_dict[(i, j)]))
                except:
                    if i in frames and j in frames:
                        #print(ced_calc_frames_frames(frecs[i], frecs[j]))
                        test_data.append((i, j, ced_calc_frames_frames(frecs[i], frecs[j])))
                    else:
                        problems.append((i,j))
                        
    # used
    import pickle
    #with open("data/combined_simplified_features.pkl","wb") as f:
    #    pickle.dump(graphs,f)
    with open("data/graphs_v1.pkl","rb") as f:
        graphs = pickle.load(f)
    
    # final data creation
    import json
    file_no = 0
    for item in train_data:
        json_data = {}
        # print(item[0])
        json_data['graph_1']=[[elem[0],elem[1]] for elem in graphs[item[0]].edges(data=True)]
        json_data['graph_2']=[[elem[0],elem[1]] for elem in graphs[item[1]].edges(data=True)]
        json_data['labels_1']=[elem[1]['label'] for elem in graphs[item[0]].nodes(data=True)]
        json_data['labels_2']=[elem[1]['label'] for elem in graphs[item[1]].nodes(data=True)]
        json_data['ged']=item[2]

        with open("data/fold_1/train_pairs/"+str(file_no)+".json", "w") as outfile:
            json.dump(json_data, outfile)
        # input("wait")
        file_no += 1
        
    # final data creation
    import json
    file_no = 0
    for item in val_data:
        json_data = {}
        # print(item[0])
        json_data['graph_1']=[[elem[0],elem[1]] for elem in graphs[item[0]].edges(data=True)]
        json_data['graph_2']=[[elem[0],elem[1]] for elem in graphs[item[1]].edges(data=True)]
        json_data['labels_1']=[elem[1]['label'] for elem in graphs[item[0]].nodes(data=True)]
        json_data['labels_2']=[elem[1]['label'] for elem in graphs[item[1]].nodes(data=True)]
        json_data['ged']=item[2]

        with open("data/fold_1/val_pairs/"+str(file_no)+".json", "w") as outfile:
            json.dump(json_data, outfile)
        # input("wait")
        file_no += 1
    
    # final data creation
    import json
    file_no = 0
    for item in test_data:
        json_data = {}
        # print(item[0])
        json_data['graph_1']=[[elem[0],elem[1]] for elem in graphs[item[0]].edges(data=True)]
        json_data['graph_2']=[[elem[0],elem[1]] for elem in graphs[item[1]].edges(data=True)]
        json_data['labels_1']=[elem[1]['label'] for elem in graphs[item[0]].nodes(data=True)]
        json_data['labels_2']=[elem[1]['label'] for elem in graphs[item[1]].nodes(data=True)]
        json_data['ged']=item[2]

        with open("data/fold_1/test_pairs/"+str(file_no)+".json", "w") as outfile:
            json.dump(json_data, outfile)
        # input("wait")
        file_no += 1

except (Exception, psycopg2.Error) as error :
    if(conn):
        print("Failed to insert record into mobile table", error)

finally:
    if(conn):
        dbcur.close()
        conn.close()

