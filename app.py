# NYC Crime Intelligence - Flask API v4
# Lance avec : python app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, pandas as pd, numpy as np, os, warnings
warnings.filterwarnings('ignore')

app  = Flask(__name__)
CORS(app)
BASE = os.path.dirname(os.path.abspath(__file__))

def load(name):
    return joblib.load(os.path.join(BASE, name))

print("Chargement des modeles...")
scaler    = load('scaler_kmeans.pkl')
kmeans    = load('kmeans_nyc.pkl')
le_group  = load('le_crime_group.pkl')
le_sev    = load('le_severity.pkl')
rf_crime  = load('rf_crime_group.pkl')
xgb_sev   = load('xgb_severity.pkl')
rf_temp   = load('rf_temporal.pkl')

try:
    rf_age    = load('rf_profiling_age.pkl')
    rf_gender = load('rf_profiling_gender.pkl')
    le_age    = load('le_profiling_age.pkl')
    HAS_PROF  = True
    print(f"Modeles profiling OK - ages : {le_age.classes_}")
except:
    HAS_PROF  = False
    print("Modeles profiling absents - stats embarquees utilisees")

print("Tous les modeles charges!")
print(f"  Crimes : {le_group.classes_}")
print(f"  Severites : {le_sev.classes_}")

FEAT_BASE = ['latitude','longitude','hour','day_of_week','month',
             'is_weekend','cluster','location_group']
FEAT_S2   = FEAT_BASE + ['crime_encoded']
FEAT_PROF = FEAT_S2   + ['sev_encoded']
FEAT_TEMP = ['hour','day_of_week','month','is_weekend','cluster','location_group']

BORO_LABELS = ['Manhattan','Brooklyn','Bronx','Queens','Staten Island']
BORO_COORDS = {
    'Manhattan':   (40.7831,-73.9712),
    'Brooklyn':    (40.6782,-73.9442),
    'Bronx':       (40.8448,-73.8648),
    'Queens':      (40.7282,-73.7949),
    'Staten Island':(40.5795,-74.1502),
}
BORO_MAP = {'Manhattan':0,'Brooklyn':1,'Bronx':2,'Queens':3,'Staten Island':4}

INTER = {
    'FELONY':      {'label':'Intervention speciale','color':'#f74f4f','cost':1200},
    'MISDEMEANOR': {'label':'Patrouille standard',  'color':'#f7a74f','cost':350},
    'VIOLATION':   {'label':'Traitement admin.',    'color':'#4ff799','cost':80},
}

PROFILE_STATS = {
    'VIOLENT':     {'age':{'<18':4,'18-24':28,'25-44':48,'45-64':18,'65+':2}, 'gender':{'Homme':88,'Femme':12}},
    'DRUG':        {'age':{'<18':3,'18-24':26,'25-44':52,'45-64':18,'65+':1}, 'gender':{'Homme':82,'Femme':18}},
    'PROPERTY':    {'age':{'<18':4,'18-24':30,'25-44':45,'45-64':20,'65+':1}, 'gender':{'Homme':70,'Femme':30}},
    'PUBLIC ORDER':{'age':{'<18':5,'18-24':25,'25-44':42,'45-64':25,'65+':3}, 'gender':{'Homme':65,'Femme':35}},
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status':'ok','city':'NYC','has_profiling':HAS_PROF,
                    'models':['kmeans','rf_crime','xgb_sev','rf_temporal']})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        d        = request.get_json()
        lat,lon  = float(d['latitude']), float(d['longitude'])
        hour     = int(d['hour'])
        dow      = int(d['day_of_week'])
        month    = int(d.get('month',6))
        is_we    = int(dow >= 5)
        loc_grp  = int(d.get('location_group',0))

        cluster  = int(kmeans.predict(scaler.transform([[lat,lon]]))[0])
        cl_lbl   = BORO_LABELS[cluster] if cluster < len(BORO_LABELS) else f'Zone {cluster}'

        x1 = pd.DataFrame([dict(latitude=lat,longitude=lon,hour=hour,day_of_week=dow,
                                 month=month,is_weekend=is_we,cluster=cluster,
                                 location_group=loc_grp)])[FEAT_BASE]
        crime_enc = int(rf_crime.predict(x1)[0])
        crime_pr  = rf_crime.predict_proba(x1)[0]
        crime_lbl = le_group.inverse_transform([crime_enc])[0]
        top3 = [{'type':le_group.inverse_transform([i])[0],'proba':round(float(crime_pr[i])*100,1)}
                for i in crime_pr.argsort()[::-1][:3]]

        x2 = x1.copy(); x2['crime_encoded'] = crime_enc
        x2 = x2[FEAT_S2]
        sev_enc  = int(xgb_sev.predict(x2)[0])
        sev_pr   = xgb_sev.predict_proba(x2)[0]
        sev_lbl  = le_sev.inverse_transform([sev_enc])[0]
        sev_conf = round(float(sev_pr.max())*100,1)
        inter    = INTER.get(sev_lbl, INTER['MISDEMEANOR'])

        return jsonify({'success':True,
            'step0':{'cluster':cluster,'label':cl_lbl},
            'step1':{'crime_group':crime_lbl,'confidence':round(float(crime_pr.max())*100,1),'top3':top3},
            'step2':{'severity':sev_lbl,'confidence':sev_conf,
                     'intervention':inter['label'],'color':inter['color'],'cost':inter['cost']}})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

@app.route('/predict_temporal', methods=['POST'])
def predict_temporal():
    try:
        d        = request.get_json()
        hour     = int(d['hour'])
        dow      = int(d['day_of_week'])
        month    = int(d.get('month',6))
        borough  = d.get('zone','Manhattan')
        loc_grp  = int(d.get('location_group',0))
        is_we    = int(dow >= 5)
        cluster  = BORO_MAP.get(borough, 0)
        lat, lon = BORO_COORDS.get(borough,(40.7128,-74.0060))

        xt = pd.DataFrame([dict(hour=hour,day_of_week=dow,month=month,
                                is_weekend=is_we,cluster=cluster,
                                location_group=loc_grp)])[FEAT_TEMP]
        crime_pr  = rf_temp.predict_proba(xt)[0]
        top_idx   = crime_pr.argsort()[::-1][:4]
        top_crimes= [{'type':le_group.inverse_transform([i])[0],'proba':round(float(crime_pr[i])*100,1)}
                     for i in top_idx]
        crime_enc = int(top_idx[0])

        x2 = pd.DataFrame([dict(latitude=lat,longitude=lon,hour=hour,day_of_week=dow,
                                 month=month,is_weekend=is_we,cluster=cluster,
                                 location_group=loc_grp,crime_encoded=crime_enc)])[FEAT_S2]
        sev_enc  = int(xgb_sev.predict(x2)[0])
        sev_lbl  = le_sev.inverse_transform([sev_enc])[0]
        sev_conf = round(float(xgb_sev.predict_proba(x2)[0].max())*100,1)
        inter    = INTER.get(sev_lbl, INTER['MISDEMEANOR'])

        return jsonify({'success':True,
            'prediction':{'top_crimes':top_crimes,'severity':sev_lbl,'confidence':sev_conf,
                          'intervention':inter['label'],'color':inter['color'],'cost':inter['cost']}})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

@app.route('/profile', methods=['POST'])
def profile():
    try:
        d        = request.get_json()
        lat, lon = float(d['latitude']), float(d['longitude'])
        hour     = int(d['hour'])
        dow      = int(d['day_of_week'])
        month    = int(d.get('month',6))
        is_we    = int(dow >= 5)
        loc_grp  = int(d.get('location_group',0))

        cluster  = int(kmeans.predict(scaler.transform([[lat,lon]]))[0])

        x1 = pd.DataFrame([dict(latitude=lat,longitude=lon,hour=hour,day_of_week=dow,
                                 month=month,is_weekend=is_we,cluster=cluster,
                                 location_group=loc_grp)])[FEAT_BASE]
        crime_enc = int(rf_crime.predict(x1)[0])
        crime_lbl = le_group.inverse_transform([crime_enc])[0]

        x2 = x1.copy(); x2['crime_encoded'] = crime_enc
        x2 = x2[FEAT_S2]
        sev_enc = int(xgb_sev.predict(x2)[0])
        sev_lbl = le_sev.inverse_transform([sev_enc])[0]

        if HAS_PROF:
            xp = x2.copy(); xp['sev_encoded'] = sev_enc
            xp = xp[FEAT_PROF]
            age_pr    = rf_age.predict_proba(xp)[0]
            age_lbl   = le_age.inverse_transform([rf_age.predict(xp)[0]])[0]
            age_all   = {le_age.inverse_transform([i])[0]:round(float(p)*100,1) for i,p in enumerate(age_pr)}
            gen_pr    = rf_gender.predict_proba(xp)[0]
            gen_lbl   = 'Femme' if rf_gender.predict(xp)[0]==1 else 'Homme'
            gen_conf  = round(float(gen_pr.max())*100,1)
            source    = 'model'
        else:
            st        = PROFILE_STATS.get(crime_lbl, PROFILE_STATS['PUBLIC ORDER'])
            age_all   = {k:float(v) for k,v in st['age'].items()}
            age_lbl   = max(st['age'], key=st['age'].get)
            gen_lbl   = max(st['gender'], key=st['gender'].get)
            gen_conf  = float(st['gender'][gen_lbl])
            source    = 'stats'

        notes = []
        if hour>=22 or hour<=5: notes.append('Crime nocturne - probabilite recidiviste augmentee')
        if sev_lbl == 'FELONY': notes.append('Crime grave (Felony) - antecedents judiciaires probables')
        if crime_lbl == 'DRUG': notes.append('Crime stupefiants - possible reseau organise')

        return jsonify({'success':True,'source':source,'crime_group':crime_lbl,'severity':sev_lbl,
            'boro': BORO_LABELS[cluster] if cluster < len(BORO_LABELS) else f'Zone {cluster}',
            'profile':{'age_predicted':age_lbl,'age_probas':age_all,
                       'gender_predicted':gen_lbl,'gender_confidence':gen_conf,
                       'context_notes':notes}})
    except Exception as e:
        return jsonify({'success':False,'error':str(e)}), 500

if __name__ == '__main__':
    print("\nAPI sur http://localhost:5000")
    app.run(debug=False, port=5000, use_reloader=False)