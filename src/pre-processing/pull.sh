# main
cd ~/Dev/smart_passage/src
mkdir "spl_data/landing"

# luggage 1
mkdir spl_data/landing/luggage_1 && cd $_
curl -L "https://universe.roboflow.com/ds/PSADr2tEHI?key=9BtUoFPgAE" > roboflow.zip && unzip roboflow.zip && rm roboflow.zip

# luggage 2
mkdir spl_data/landing/luggage_2 && cd $_
curl -L "https://universe.roboflow.com/ds/6X4cgAyu0M?key=SSwSEX01BX" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# wheelchair 1
mkdir spl_data/landing/wheelchair_1 && cd $_
curl -L "https://universe.roboflow.com/ds/MKANDzD0y1?key=OkBslMWHWb" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# wheelchair 2
mkdir spl_data/landing/wheelchair_2 && cd $_
# https://universe.roboflow.com/wheelchair-8z6vo/wheelchair1-pwebi/dataset/1
curl -L "https://app.roboflow.com/ds/7kpBQRO3hR?key=dLNkKoia4z" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# wheelchair 3
mkdir spl_data/landing/wheelchair_3 && cd $_
curl -L "https://universe.roboflow.com/ds/Oehj92YfU3?key=D7W2xxRsw5" > roboflow.zip; unzip roboflow.zip && rm roboflow.zip

# box2
mkdir spl_data/landing/box_2 && cd $_
curl -L "https://universe.roboflow.com/ds/Ylz1a00hua?key=TG56cTyl5c" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip

# box3
mkdir spl_data/landing/box_3 && cd $_
curl -L "https://universe.roboflow.com/ds/nIaZ6hvskn?key=SIfTs7MbQF" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip