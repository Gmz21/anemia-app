import streamlit as st
import pandas as pd
import networkx as nx
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import folium_static
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import numpy as np

# Funciones

# Función para cargar los datos del CSV
@st.cache_data
def cargar_datos(ruta):
    df = pd.read_csv(ruta, delimiter=';')
    # Segmentación de datos con pandas
    df = df[(df['FECHA_REGISTRO'] > 20230720) & 
            (df['EDAD_REGISTRO'] >= 3) & 
            (df['EDAD_REGISTRO'] <= 10) & 
            (df['TIPO_EDAD'] == 'A')]
    return df.reset_index(drop=True)

# Función para calcular distancias entre viviendas con vecinos más cercanos
@st.cache_data
def calcular_distancias_viviendas(df, k_vecinos=25):
    coords = np.radians(df[['LATITUD', 'LONGITUD']].to_numpy())
    tree = BallTree(coords, metric='haversine')
    k = min(k_vecinos + 1, len(coords))
    distancias, indices = tree.query(coords, k=k)

    dist_km = distancias[:, 1:] * 6371
    indices_k = indices[:, 1:]

    idx_origen = np.repeat(np.arange(len(df)), indices_k.shape[1])
    idx_destino = indices_k.reshape(-1)
    dist_flat = dist_km.reshape(-1)

    pk = df['PK_REGISTRO'].to_numpy()
    pk1 = pk[idx_origen]
    pk2 = pk[idx_destino]
    pk_min = np.minimum(pk1, pk2)
    pk_max = np.maximum(pk1, pk2)

    df_distancias = pd.DataFrame({
        'PK_1': pk_min,
        'PK_2': pk_max,
        'DISTANCIA_KM': np.round(dist_flat, 3)
    })
    df_distancias = df_distancias.drop_duplicates(['PK_1', 'PK_2']).reset_index(drop=True)

    return df_distancias

# Función para conectar nodos de manera específica
def conectar_nodos(n1, n2, df, grafo):
        coord1 = tuple(df.iloc[n1][['LATITUD', 'LONGITUD']])
        coord2 = tuple(df.iloc[n2][['LATITUD', 'LONGITUD']])

        # Se calcula la distancia entre las coordenadas
        distancia_km = geodesic(coord1, coord2).km

        # Se agrega la arista al grafo con el peso de la distancia
        grafo.add_edge(n1, n2, weight=round(distancia_km, 3))

# Función para construir el grafo a partir de los dataframes
@st.cache_resource
def construir_grafo(df_anemia, df_distancia):
    G = nx.Graph()
    colores = []
    pk_to_index = df_anemia.reset_index()[['index', 'PK_REGISTRO']].set_index('PK_REGISTRO')['index'].to_dict()

    # Agregar nodos con coordenadas, el código de registro y creamos el arreglo de colores
    for idx, row in df_anemia.iterrows():
        G.add_node(idx, pos_lat=row['LATITUD'], pos_lon=row['LONGITUD'], cod=row['PK_REGISTRO'])
        if row['GRADO_SEVERIDAD'] == 'LEV':
            colores.append('green')
        elif row['GRADO_SEVERIDAD'] == 'MOD':
            colores.append('orange')
        elif row['GRADO_SEVERIDAD'] == 'SEV':
            colores.append('red')

    # Conectar nodos según las distancias
    for _, row in df_distancia.iterrows():
        pk1 = row['PK_1']
        pk2 = row['PK_2']
        dist = row['DISTANCIA_KM']

        idx1 = pk_to_index.get(pk1)
        idx2 = pk_to_index.get(pk2)

        if idx1 is not None and idx2 is not None:
            G.add_edge(idx1, idx2, weight=dist)    

    # Conectar nodos específicos
    conectar_nodos(1494, 447, df_anemia, G)
    conectar_nodos(553, 1494, df_anemia, G)

        
    return G, colores

# Función que genera el mapa inicial
@st.cache_resource
def generar_mapa(_G, colores, df):
    mapa = folium.Map(location=[-6.5, -76.5], zoom_start=8)
    marker_cluster = MarkerCluster().add_to(mapa)

    for n, d in _G.nodes(data=True):
        if n < len(df):
            folium.CircleMarker(
                location=(d['pos_lat'], d['pos_lon']),
                radius=4,
                color=colores[n],
                fill=True,
                fill_color=colores[n],
                fill_opacity=0.9,
                tooltip=f"Paciente Nº {n}<br>Edad: {df.iloc[n]['EDAD_REGISTRO']}<br>Severidad: {df.iloc[n]['GRADO_SEVERIDAD']}<br>Hospital: {df.iloc[n]['NOMBRE_ESTABLECIMIENTO']}"
            ).add_to(marker_cluster)
        else:
            folium.Marker(
                location=(d['pos_lat'], d['pos_lon']),
                icon=folium.Icon(color='blue', icon='plus-sign'),
                tooltip=f"Hospital: {d['cod']}"
            ).add_to(mapa)

    return mapa

def generar_mapa_ruta(G, colores, nodos_ruta, df):

    # Verifica si la ruta está vacía
    if not nodos_ruta:
        return folium.Map(location=[-6.5, -76.5], zoom_start=8)

    # Inicializa el mapa centrado en el primer nodo
    lat, lon = G.nodes[nodos_ruta[0]]['pos_lat'], G.nodes[nodos_ruta[0]]['pos_lon']
    mapa = folium.Map(location=[lat, lon], zoom_start=8)

    # Agregar los nodos del camino/ruta, se usará un marcador para su representación
    for n in nodos_ruta:
        data = G.nodes[n]
        if n < len(df):
            folium.Marker(
                location=(data['pos_lat'], data['pos_lon']),
                icon=folium.Icon(color=colores[n]),
                tooltip=f"Paciente Nº {n}<br>Edad: {df.iloc[n]['EDAD_REGISTRO']}<br>Severidad: {df.iloc[n]['GRADO_SEVERIDAD']}<br>Hospital: {df.iloc[n]['NOMBRE_ESTABLECIMIENTO']}"
            ).add_to(mapa)
        else:
            folium.Marker(
                location=(data['pos_lat'], data['pos_lon']),
                icon=folium.Icon(color='blue', icon='plus-sign'),
                tooltip=f"Hospital: {data['cod']}"
            ).add_to(mapa)

    # Se dibujan las aristas de la ruta
    for i in range(len(nodos_ruta)-1):
        n1, n2 = nodos_ruta[i], nodos_ruta[i+1]
        loc1 = (G.nodes[n1]['pos_lat'], G.nodes[n1]['pos_lon'])
        loc2 = (G.nodes[n2]['pos_lat'], G.nodes[n2]['pos_lon'])
        folium.PolyLine([loc1, loc2], color='blue', weight=3).add_to(mapa)

    return mapa

def agregar_hospitales_al_grafo(G, colores, df_anemia, df_hospitales, k_vecinos=15):
    offset = len(G.nodes)
    coords_anemia = np.radians(df_anemia[['LATITUD', 'LONGITUD']].values)
    tree = BallTree(coords_anemia, metric='haversine')
    
    for i, row in df_hospitales.iterrows():
        idx = offset + i
        lat, lon = row['LATITUD'], row['LONGITUD']
        nombre = row['NOMBRE']
        G.add_node(idx, pos_lat=lat, pos_lon=lon, cod=nombre)
        colores.append('blue')

        coord_hosp = np.radians([[lat, lon]])
        distancias, indices = tree.query(coord_hosp, k=min(k_vecinos, len(coords_anemia)))

        for dist, index in zip(distancias[0], indices[0]):
            distancia_km = dist * 6371
            G.add_edge(idx, index, weight=round(distancia_km, 3))

@st.cache_resource
def construir_grafo_completo(df_anemia, df_hospitales, k_vecinos_viviendas=25, k_vecinos_hospitales=15):
    df_distancias = calcular_distancias_viviendas(df_anemia, k_vecinos=k_vecinos_viviendas)
    G, colores = construir_grafo(df_anemia, df_distancias)
    agregar_hospitales_al_grafo(G, colores, df_anemia, df_hospitales, k_vecinos=k_vecinos_hospitales)
    return G, colores


# Página de Streamlit

# Hospitales desde los que se reparte la galleta
df_hospitales = pd.DataFrame({
    'NOMBRE': ['NUEVE DE ABRIL', 'HOSPITAL LAMAS', 'BUENOS AIRES'],
    'LATITUD': [-6.48597333, -6.41549003, -5.91518],
    'LONGITUD': [-76.372355, -76.51878761, -77.079845]
})



# Inicialización de variables
ruta = "ANEMIA_DA.csv"
Data_Anemia = cargar_datos(ruta)
G, color = construir_grafo_completo(Data_Anemia, df_hospitales)
mapa = generar_mapa(G, color, Data_Anemia)

# Interfaz de usuario
st.title("Anemia en el departamento de San Martín")

st.subheader("""Gianmarco Fabian Jiménez Guerra""")

st.subheader("Mapa de conexiones geográficas de pacientes")
folium_static(mapa)

# Selección de opciones de algoritmos
algoritmo = st.selectbox("Selecciona el algoritmo: ", ['Dijkstra', 'Kruskal'])

if algoritmo == 'Dijkstra':
    # Nodo origen y destino
    nodos_hospital = {G.nodes[n]['cod']: n for n in G.nodes if n >= len(Data_Anemia)}
    hospital_nombre = st.selectbox("Selecciona el hospital", list(nodos_hospital.keys()))
    nodo_origen = nodos_hospital[hospital_nombre]
    destino = st.number_input("Paciente destino", min_value=0, max_value=len(Data_Anemia)-1, step=1)

    if st.button("Calcular ruta más corta"):
        try:
            # Calcular la ruta más corta usando Dijkstra
            ruta_dijkstra = nx.dijkstra_path(G, source=nodo_origen, target=destino)
            mapa_ruta = generar_mapa_ruta(G, color, ruta_dijkstra, Data_Anemia)
            folium_static(mapa_ruta)
            distancia_total = sum(G.edges[ruta_dijkstra[i], ruta_dijkstra[i+1]]['weight'] for i in range(len(ruta_dijkstra)-1))
            st.success(f"Distancia recorrida: {round(distancia_total, 2)} km")
        except Exception as e:
            # Manejo de error
            st.error(f"No es posible calcular la ruta más corta: {e}")

elif algoritmo == 'Kruskal':
    nodos_hospital = {G.nodes[n]['cod']: n for n in G.nodes if n >= len(Data_Anemia)}
    hospital_nombre = st.selectbox("Selecciona el hospital", list(nodos_hospital.keys()))
    nodos_hospital = nodos_hospital[hospital_nombre]
    if st.button("Calcular MST"):
        try:
            # Filtrar los nodos de casos severos        
            nodos_severos = [
            n for n in range(len(Data_Anemia))
            if Data_Anemia.iloc[n]['GRADO_SEVERIDAD'] == 'SEV'
            ]

            nodos_severos.append(nodos_hospital)
            # Se crea un grafo con los nodos severos
            grafo_severos = nx.Graph()
            for n in nodos_severos:
                grafo_severos.add_node(
                    n,
                    pos_lat=G.nodes[n]['pos_lat'],
                    pos_lon=G.nodes[n]['pos_lon']
                )

            # Se conectan todos los nodos entre sí
            for i in range(len(nodos_severos)):
                for j in range(i+1, len(nodos_severos)):
                    n1, n2 = nodos_severos[i], nodos_severos[j]
                    coord1 = (G.nodes[n1]['pos_lat'], G.nodes[n1]['pos_lon'])
                    coord2 = (G.nodes[n2]['pos_lat'], G.nodes[n2]['pos_lon'])
                    distancia_km = geodesic(coord1, coord2).km
                    grafo_severos.add_edge(n1, n2, weight=round(distancia_km, 3))

            # Se calcula el MST usando Kruskal
            mst = nx.minimum_spanning_tree(grafo_severos)
            nodos_mst = list(mst.nodes)
            edges_mst = list(mst.edges)

            if nodos_mst:
                lat0 = grafo_severos.nodes[nodos_mst[0]]['pos_lat']
                lon0 = grafo_severos.nodes[nodos_mst[0]]['pos_lon']
                mapa_mst = folium.Map(location=[lat0, lon0], zoom_start=8)

                # Se muestra cada nodo del MST
                for n in nodos_mst:
                    data = grafo_severos.nodes[n]
                    tooltip = ""
                    if n < len(Data_Anemia):
                        tooltip = f"Paciente Nº {n}<br>Edad: {Data_Anemia.iloc[n]['EDAD_REGISTRO']}<br>Severidad: {Data_Anemia.iloc[n]['GRADO_SEVERIDAD']}<br>Hospital: {Data_Anemia.iloc[n]['NOMBRE_ESTABLECIMIENTO']}"
                        icon = folium.Icon(color=color[n])
                    else:
                        tooltip = f"Hospital: {G.nodes[n]['cod']}"
                        icon = folium.Icon(color='blue', icon='plus-sign')
                    
                    folium.Marker(
                        location=(data['pos_lat'], data['pos_lon']),
                        icon=icon,
                        tooltip=tooltip
                    ).add_to(mapa_mst)

                # Se dibujan las aristas del MST
                for n1, n2 in edges_mst:
                    loc1 = (grafo_severos.nodes[n1]['pos_lat'], grafo_severos.nodes[n1]['pos_lon'])
                    loc2 = (grafo_severos.nodes[n2]['pos_lat'], grafo_severos.nodes[n2]['pos_lon'])
                    folium.PolyLine([loc1, loc2], color='blue', weight=3).add_to(mapa_mst)

                folium_static(mapa_mst)

                longitud_total_mst = sum(mst.edges[e]['weight'] for e in mst.edges)
                st.success(f"Longitud total del MST: {round(longitud_total_mst, 2)} km")

            else:
                st.warning("El MST no contiene nodos.")
        except Exception as e:
            st.warning(f"Error al calcular el MST: {e}")


