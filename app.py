import math
import dash_bootstrap_components as dbc
import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import shapely
from dash import Dash, html, dcc
from dash.dependencies import Output, Input
from dash.exceptions import PreventUpdate
from geopy.distance import geodesic
from plotly import graph_objects as go
from sklearn.cluster import KMeans

# CONSTANT VALUES
zoom = 10.0  # Map zoom
latInitial = 22.628  # Center of latitude
lonInitial = 114.055  # Center of longitude
k_value = 5  # Initial value of k

# DATA
# Shenzhen grid data
grid = gpd.GeoDataFrame.from_file(r'resources/szgrid/grid_sz.shp', encoding='UTF-8')
grid.crs = 'EPSG:4326'
sz = gpd.GeoDataFrame.from_file(r'resources/szosm/sz.shp', encoding='UTF-8')

# Taxi GPS data
taxi_data = pd.read_csv(r'resources/taxi_dataset_sample.csv', encoding='UTF-8_sig', header=None, names=['VehicleNum', 'Stime', 'Lng', 'Lat', 'OpenStatus', 'Speed'],
                        dtype={'VehicleNum': 'int64', 'Stime': 'string', 'Lng': 'float64', 'Lat': 'float64'})
taxi_data = taxi_data.sort_values(by=['VehicleNum', 'Stime'])
taxi_data = taxi_data[(taxi_data['Lng'] > 113.65) & (taxi_data['Lng'] < 114.55) & (taxi_data['Lat'] > 22.45) & (taxi_data['Lat'] < 22.85)]
# Taxi OD Extraction
taxi_OD_data_overview = taxi_data[-((taxi_data['OpenStatus'].shift(-1) == taxi_data['OpenStatus'].shift()) &
                                    (taxi_data['OpenStatus'].shift(-1) != taxi_data['OpenStatus']) &
                                    (taxi_data['VehicleNum'].shift(-1) == taxi_data['VehicleNum'].shift()) &
                                    (taxi_data['VehicleNum'].shift(-1) == taxi_data['VehicleNum']))]
taxi_OD_data_overview['StatusChange'] = taxi_OD_data_overview['OpenStatus'] - taxi_OD_data_overview['OpenStatus'].shift()
taxi_OD_data_overview = taxi_OD_data_overview[((taxi_OD_data_overview['StatusChange'] == 1) | (taxi_OD_data_overview['StatusChange'] == -1))]
taxi_OD_data_overview = taxi_OD_data_overview[['VehicleNum', 'Stime', 'Lng', 'Lat', 'StatusChange']]
taxi_OD_data_overview.columns = ['VehicleNum', 'Stime', 'SLng', 'SLat', 'StatusChange']
taxi_OD_data_overview['ELng'] = taxi_OD_data_overview['SLng'].shift(-1)
taxi_OD_data_overview['ELat'] = taxi_OD_data_overview['SLat'].shift(-1)
taxi_OD_data_overview['Etime'] = taxi_OD_data_overview['Stime'].shift(-1)
taxi_OD_data_overview = taxi_OD_data_overview[taxi_OD_data_overview['StatusChange'] == 1]
# Extracting Pick-up and Drop-off Blocks
pickup = taxi_OD_data_overview[['SLng', 'SLat']]
pickup['geometry'] = gpd.points_from_xy(pickup.SLng, pickup.SLat)
pickup = gpd.GeoDataFrame(pickup, geometry=pickup['geometry'], crs=4326)
pickup_result = gpd.sjoin(grid, pickup, op='intersects').groupby(['LngCol', 'LatCol']).size().reset_index(name='count')
pickup_result = pd.merge(grid, pickup_result, on=['LngCol', 'LatCol'], how='left')
pickup_result['count'] = pickup_result['count'].fillna(0)
dropoff = taxi_OD_data_overview[['ELng', 'ELat']]
dropoff['geometry'] = gpd.points_from_xy(dropoff.ELng, dropoff.ELat)
dropoff = gpd.GeoDataFrame(dropoff, geometry=dropoff['geometry'], crs=4326)
dropoff_result = gpd.sjoin(grid, dropoff, op='intersects').groupby(['LngCol', 'LatCol']).size().reset_index(name='count')
dropoff_result = pd.merge(grid, dropoff_result, on=['LngCol', 'LatCol'], how='left')
dropoff_result['count'] = dropoff_result['count'].fillna(0)
# Order duration and hour-variant accumulation
taxi_OD_data_overview = taxi_OD_data_overview[['VehicleNum', 'Stime', 'SLng', 'SLat', 'ELng', 'ELat', 'Etime']]
taxi_OD_data_overview['duration'] = pd.to_datetime(taxi_OD_data_overview['Etime']) - pd.to_datetime(taxi_OD_data_overview['Stime'])
taxi_OD_data_overview['duration'] = taxi_OD_data_overview['duration'].apply(lambda r: r.seconds)
taxi_OD_data_overview['Hour'] = taxi_OD_data_overview.apply(lambda r: r['Stime'][:2], axis=1).astype(int)
hour_count = taxi_OD_data_overview.groupby('Hour')['VehicleNum'].count()
hour_count = hour_count.rename('count').reset_index()

# SVD Operation, re-calculate grids
lat1 = 22.391837
lat2 = 22.908748
lonStart = 113.73094
latStart = 22.387837
accuracy = 500
deltaLon = accuracy * 360 / (2 * math.pi * 6371004 * math.cos((lat1 + lat2) * math.pi / 360))
deltaLat = accuracy * 360 / (2 * math.pi * 6371004)
taxi_OD_data_overview['SLONCOL'] = ((taxi_OD_data_overview['SLng'] - (lonStart - deltaLon / 2)) / deltaLon).astype(int)
taxi_OD_data_overview['SLATCOL'] = ((taxi_OD_data_overview['SLat'] - (latStart - deltaLat / 2)) / deltaLat).astype(int)
Gridcount = taxi_OD_data_overview.groupby(['Hour', 'SLONCOL', 'SLATCOL'])['VehicleNum'].count().reset_index()
Gridcount['grid_id'] = Gridcount['SLONCOL'].astype(str) + '|' + Gridcount['SLATCOL'].astype(str)
Gridmatrix = Gridcount.pivot(index='Hour', columns='grid_id', values='VehicleNum')
nulldays = Gridmatrix.isnull().sum()
retaindays = nulldays[nulldays < 18].index  # Filtering all grids with at least 18 hours of operation
Gridmatrix = Gridmatrix[retaindays].fillna(0)  # Filling all NaN grids in matrix with 0
M = Gridmatrix.values
U, sigma, VT = np.linalg.svd(M)
for i in range(U.shape[0]):
    ui = U[:, i]
    flag = np.sign(ui[abs(ui) == abs(ui).max()])
    U[:, i] = flag * U[:, i]
    VT[i, :] = flag * VT[i, :]

# Shenzhen road network shapefile data
sz_road_network_selected = gpd.read_file(r'resources/shenzhen_road_selected/shenzhen_road_selected.shp', encoding='UTF-8')
sz_road_network_selected = sz_road_network_selected.to_crs(epsg=4326)
# Constructing service area
road_service_area = sz_road_network_selected.buffer(distance=0.0001, resolution=16)
taxi_data['geometry'] = gpd.points_from_xy(taxi_data.Lng, taxi_data.Lat)
taxi_gpd = gpd.GeoDataFrame(taxi_data['geometry'], geometry=taxi_data['geometry'], crs=4326)  # Intersecting must apply geoDataFrame
density = np.empty(0)  # Defined as the number of data points intersected with the selected road section
for index, line in road_service_area.items():
    road_object = road_service_area.iloc[index:index + 1]
    density = np.append(density, taxi_gpd[taxi_gpd.intersects(road_object.unary_union)].size)
sz_road_network_selected['density'] = density
# Constructing graphs for road network
lats = np.empty(0)
lons = np.empty(0)
names = np.empty(0, dtype=object)
intersections = np.empty(0, dtype=object)
for feature, intersection1, intersection2, name in zip(sz_road_network_selected.geometry, sz_road_network_selected.Intersec1, sz_road_network_selected.Intersec2, sz_road_network_selected.Road):
    if isinstance(feature, shapely.geometry.linestring.LineString):
        linestrings = [feature]
    elif isinstance(feature, shapely.geometry.multilinestring.MultiLineString):
        linestrings = feature.geoms
    else:
        continue
    for linestring in linestrings:
        x, y = linestring.xy
        lats = np.append(lats, y)
        lons = np.append(lons, x)
        names = np.append(names, [name] * 2)
        intersections = np.append(intersections, intersection1)
        intersections = np.append(intersections, intersection2)
        # Avoid connection between different roads, ensuring separated road sections
        lats = np.append(lats, None)
        lons = np.append(lons, None)
        names = np.append(names, None)
        intersections = np.append(intersections, None)

shenzhen_road = pd.DataFrame(columns=['Lng', 'Lat', 'Road', 'Intersection'])
shenzhen_road['Lat'] = lats
shenzhen_road['Lng'] = lons
shenzhen_road['Road'] = names
shenzhen_road['Intersection'] = intersections

# Metro station data
station = pd.read_csv(r'resources/shenzhen_metro/shenzhen_metro_stations.csv', encoding='GBK')
station['geometry'] = gpd.points_from_xy(x=station['Lng'], y=station['Lat'])
station = gpd.GeoDataFrame(station, geometry=station['geometry'], crs=4326)
# Establishing metro network by extracting metro 'OD', connecting closing stations
station['Line1'] = station['Line'].shift(-1)
station['Station1'] = station['Station'].shift(-1)
station['Lng1'] = station['Lng'].shift(-1)
station['Lat1'] = station['Lat'].shift(-1)
station = station[station['Line'] == station['Line1']]
station['OStation'] = station['Line'] + station['Station']
station['DStation'] = station['Line1'] + station['Station1']
# Calculating distances between metro stations (direct line, large error)
stations_distance = np.empty(0)
for r in range(len(station)):
    length = geodesic((station.iloc[r]['Lat'], station.iloc[r]['Lng']), (station.iloc[r]['Lat1'], station.iloc[r]['Lng1'])).km
    stations_distance = np.append(stations_distance, length)
station['length'] = stations_distance
# Dividing: normal stations (only one line through) and transfer stations (more than one line through)
edge = station[['Lng', 'Lat', 'OStation', 'Lng1', 'Lat1', 'DStation', 'length']]
edge.loc[:, 'duration'] = edge.loc[:, 'length'] * 1.2
edge.columns = ['OLng', 'OLat', 'OStation', 'DLng', 'DLat', 'DStation', 'length', 'duration']
# Reloading the station data, saving memory
station = pd.read_csv(r'resources/shenzhen_metro/shenzhen_metro_stations.csv', encoding='GBK')
station['stop'] = station['Line'] + station['Station']
transfer_station = station.groupby(['Station'])['Line'].count().rename('count').reset_index()
transfer_station = pd.merge(station, transfer_station[transfer_station['count'] > 1]['Station'], on='Station')
transfer_station = transfer_station[['Lng', 'Lat', 'Station', 'Line', 'stop']].drop_duplicates()
transfer_station = pd.merge(transfer_station, transfer_station, on='Station')
edge2 = transfer_station[transfer_station['Line_x'] != transfer_station['Line_y']][['Lng_x', 'Lat_x', 'stop_x', 'Lng_y', 'Lat_y', 'stop_y']]
edge2.columns = ['OLng', 'OLat', 'OStation', 'DLng', 'DLat', 'DStation']
edge2['duration'] = 15
edge3 = pd.concat([edge, edge2]).drop_duplicates()

# Constructing metro network
metro_station_Lngs = np.empty(0)
metro_station_Lats = np.empty(0)
metro_station_names = np.empty(0)
metro_lines = np.empty(0)
metro_station_names_with_line = np.empty(0)
grouped_lines = station.groupby('Line')
for line, data in grouped_lines:
    metro_station_Lngs = np.append(metro_station_Lngs, data['Lng'])
    metro_station_Lngs = np.append(metro_station_Lngs, None)
    metro_station_Lats = np.append(metro_station_Lats, data['Lat'])
    metro_station_Lats = np.append(metro_station_Lats, None)
    metro_station_names = np.append(metro_station_names, data['Station'])
    metro_station_names = np.append(metro_station_names, None)
    metro_lines = np.append(metro_lines, data['Line'])
    metro_lines = np.append(metro_lines, None)
    metro_station_names_with_line = np.append(metro_station_names_with_line, data['stop'])
    metro_station_names_with_line = np.append(metro_station_names_with_line, None)

metro_network = pd.DataFrame(data=[metro_station_Lngs, metro_station_Lats, metro_station_names, metro_lines, metro_station_names_with_line], index=['Lngs', 'Lats', 'Names', 'Lines', 'Stops']).T

# Metro Smart Card Data
smart_cardID_data = pd.read_csv(r'resources/Smartcard_dataset_sample.csv', encoding='GBK', header=None, names=['SmartCardID', 'Time', 'TransactionType', 'Station'])
smart_cardID_data = smart_cardID_data[smart_cardID_data['TransactionType'] != 31]  # Only metro data
smart_cardID_data = smart_cardID_data[smart_cardID_data['Station'] != 'None']  # Filtering stations with names
smart_cardID_data['Hour'] = smart_cardID_data['Time'].apply(lambda r: r.split(':')[0])
smart_cardID_data = smart_cardID_data[smart_cardID_data['Time'].apply(lambda r: int(r.split(':')[0])) > 6]
smart_cardID_data['Time'] = smart_cardID_data['Time'].apply(lambda r: r.split(':')[0] + ':' + r.split(':')[1][0] + '0')
smart_cardID_data = smart_cardID_data.sort_values(by='Time')
smart_cardID_data['TransactionType'] = smart_cardID_data['TransactionType'].map({21: 'Swipe-In', 22: 'Swipe-Out'})  # Mapping operations to digits
metro_data_by_hour = smart_cardID_data.groupby(by=['Hour', 'Station', 'TransactionType'])['SmartCardID'].count().rename('count').reset_index()
smart_cardID_data = smart_cardID_data.merge(station, on='Station', how='inner').drop_duplicates()  # Eliminate repeating stations


# Extracting peak stations in the morning and in the evening
def metro_peak_extract(start, end):
    metro_data_peak = smart_cardID_data[(smart_cardID_data['Hour'].apply(lambda r: int(r)) <= end) & (smart_cardID_data['Hour'].apply(lambda r: int(r) >= start))].groupby(by=['TransactionType', 'Station'])[
        'SmartCardID'].count().rename('count').reset_index()
    pt = pd.pivot_table(metro_data_peak, values='count', index='Station', columns='TransactionType', aggfunc='sum')
    filter_In = (pt['Swipe-In'] > pt['Swipe-Out'])
    filter_Out = (pt['Swipe-Out'] > pt['Swipe-In'])
    peak_In = pt[filter_In].reset_index()
    peak_In = peak_In[['Station', 'Swipe-In']]
    peak_In.reset_index(drop=True)
    peak_In.columns = ['Station', 'count']
    peak_In['Type'] = 'Swipe-In'
    peak_Out = pt[filter_Out].reset_index()
    peak_Out = peak_Out[['Station', 'Swipe-Out']]
    peak_Out.reset_index(drop=True)
    peak_Out.columns = ['Station', 'count']
    peak_Out['Type'] = 'Swipe-Out'
    peak = pd.concat([peak_In, peak_Out], ignore_index=True)
    return peak


# Setting time periods of morning and evening peaks
morning_peak = metro_peak_extract(7, 9)
evening_peak = metro_peak_extract(17, 19)

# Region based accumulation of metro data
metro_data_by_region = smart_cardID_data.groupby(['Region', 'TransactionType', 'Hour'])['Station'].count().rename('count').reset_index()

# Call Detail Record Data
CDR_data = pd.read_csv(r'resources/CDR_dataset_sample.csv', encoding='UTF-8_sig', header=None, names=['SIM Card ID', 'Time', 'Lng', 'Lat'])
data_to_cluster = CDR_data.groupby(by=['Lng', 'Lat'])['SIM Card ID'].count().rename('count').reset_index()
data_to_cluster_zip = np.array(list(zip(data_to_cluster['Lng'].to_list(), data_to_cluster['Lat'].to_list())))
CDR_data['Hour'] = CDR_data['Time'].apply(lambda r: int(r.split(':')[0]))
CDR_data = CDR_data[['SIM Card ID', 'Hour', 'Lng', 'Lat']]
CDR_data = CDR_data.sort_values(by=['Hour'])
CDR_data = CDR_data.groupby(by=['Hour', 'Lng', 'Lat'])['SIM Card ID'].count().rename('count').reset_index()

# -------------------------------------------------------Figures
order_overview = go.Figure(
    data=[go.Bar(
        x=hour_count['Hour'],
        y=hour_count['count']
    )],
    layout=go.Layout(
        autosize=True,
        margin=go.layout.Margin(l=20, r=20, t=50, b=40),
        title='The Order Overview of Taxis',
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )
)
order_overview.update_xaxes(tick0=0, dtick=1)
order_overview.update_layout(
    xaxis=dict(title='Hour'),
    yaxis=dict(title='Number of Orders'),
    autosize=True
)
duration_overview = go.Figure(
    data=[
        go.Box(
            x=taxi_OD_data_overview['Hour'],
            y=taxi_OD_data_overview['duration'] / 60,
            boxmean='sd',  # represent mean and standard deviation
            quartilemethod="exclusive"
        )
    ],
    layout=go.Layout(
        autosize=True,
        margin=go.layout.Margin(l=20, r=20, t=50, b=40),
        title='The Duration Overview of Taxis',
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )
)
duration_overview.update_xaxes(tick0=0, dtick=1)  # axis starting from 0 by 1
duration_overview.update_yaxes(range=[0, 100])
duration_overview.update_layout(
    xaxis=dict(
        title='Hour'
    ),
    yaxis=dict(
        title='Duration (minute)'
    ),
)

pickup_block = px.choropleth_mapbox(pickup_result, geojson=pickup_result.geometry, locations=pickup_result.index, color='count', opacity=0.4)
pickup_block.update_layout(
    title='pick-up block',
    mapbox={
        'center': {'lon': lonInitial, 'lat': latInitial},
        'style': 'open-street-map',
        'zoom': 9},
    autosize=True,
    margin=go.layout.Margin(l=10, r=30, t=50, b=30),
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    )
)
dropoff_block = px.choropleth_mapbox(dropoff_result, geojson=dropoff_result.geometry, locations=dropoff_result.index, color='count', opacity=0.4)
dropoff_block.update_layout(
    title='drop-off block',
    mapbox={
        'center': {'lon': lonInitial, 'lat': latInitial},
        'style': 'open-street-map',
        'zoom': 9},
    autosize=True,
    margin=go.layout.Margin(l=10, r=30, t=50, b=30),
    hoverlabel=dict(
        bgcolor="white",
        font_size=16,
        font_family="Rockwell"
    )
)

fig_map_metro_stations = px.scatter_mapbox(station, lat='Lat', lon='Lng', color='Region')
fig_map_metro_stations.update_layout(
    title='The Distribution of Metro Stations in Shenzhen',
    autosize=True,
    margin=go.layout.Margin(l=10, r=10, t=40, b=20),
    hoverlabel=dict(
        font_size=16,
        font_family="Rockwell"
    ),
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(255,255,255,0.4)'
    ),
    mapbox={
        'center': {'lon': lonInitial, 'lat': latInitial},
        'style': 'open-street-map',
        'zoom': zoom - 0.5
    }
)
fig_metro_flow_by_station = px.histogram(metro_data_by_hour.sort_values(['Hour', 'count']), x='Station', y='count', color='TransactionType', animation_frame='Hour',
                                         color_discrete_map={'Swipe-In': 'royalblue', 'Swipe-Out': 'firebrick'})
fig_metro_flow_by_station.update_layout(
    xaxis=dict(title='Station', tickfont=dict(size=8), tickangle=60),
    yaxis=dict(title='Count'),
    autosize=True,
    title='The Metro Traffic Flow of Shenzhen by Station',
    margin=go.layout.Margin(l=10, r=10, t=40, b=20),
    bargap=0.1,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(0,0,0,0)'
    ),
    hoverlabel=dict(
        font_size=16,
        font_family="Rockwell"
    ),
    transition_duration=500,
    updatemenus=[dict(type='buttons', showactive=True, buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 1000, 'redraw': True}, 'transition': {'duration': 1000, 'easing': 'quadratic-in-out'}}]),
                                                                dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}])])]
)

metro_flow_peak_morning = px.sunburst(morning_peak, path=['Type', 'Station'], values='count', color_continuous_scale='turbid', color='count', title='Morning Peak')
metro_flow_peak_evening = px.sunburst(evening_peak, path=['Type', 'Station'], values='count', color_continuous_scale='turbid', color='count', title='Evening Peak')

fig_metro_flow_by_region = px.histogram(metro_data_by_region, x='Region', y='count', color='TransactionType', animation_frame='Hour', barmode='group', color_discrete_map={'Swipe-In': 'royalblue', 'Swipe-Out': 'firebrick'})
fig_metro_flow_by_region.update_yaxes(range=[0, 50000])
fig_metro_flow_by_region.update_layout(
    xaxis=dict(title='Region'),
    yaxis=dict(title='Count'),
    title='The Metro Traffic Flow of Shenzhen by Region',
    autosize=True,
    margin=go.layout.Margin(l=10, r=10, t=40, b=20),
    bargap=0.1,
    legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        bgcolor='rgba(0,0,0,0)'
    ),
    hoverlabel=dict(
        font_size=16,
        font_family="Rockwell"
    ),
    transition_duration=500,
    updatemenus=[dict(type='buttons', showactive=True, buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 1000, 'redraw': True}, 'transition': {'duration': 1000, 'easing': 'quadratic-in-out'}}]),
                                                                dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}])])]
)
# Graph Structure for route planning
# graph for Road Network Route Planning
starts_lng = sz_road_network_selected.geometry.apply(lambda x: x.coords[0][1])
starts_lat = sz_road_network_selected.geometry.apply(lambda x: x.coords[0][0])
ends_lng = sz_road_network_selected.geometry.apply(lambda x: x.coords[-1][1])
ends_lat = sz_road_network_selected.geometry.apply(lambda x: x.coords[-1][0])

shenzhen_road_graph_structure = pd.DataFrame({
    'starts_lng': starts_lng,
    'starts_lat': starts_lat,
    'ends_lng': ends_lng,
    'ends_lat': ends_lat,
    'starts_id': sz_road_network_selected['Intersec1'] + ' with ' + sz_road_network_selected['Road'],
    'ends_id': sz_road_network_selected['Intersec2'] + ' with ' + sz_road_network_selected['Road'],
    'id': sz_road_network_selected['Intersec1'] + ',' + sz_road_network_selected['Intersec2'] + ',' + sz_road_network_selected['Road'],
    'volume': sz_road_network_selected['density']
})
G_automobile = nx.Graph()
for i, row in shenzhen_road_graph_structure.iterrows():
    node1 = (row['starts_lng'], row['starts_lat'])
    node2 = (row['ends_lng'], row['ends_lat'])
    G_automobile.add_edge(node1, node2)

for u, v in G_automobile.edges():
    length = geodesic(u, v).km  # Calculating distances between intersections in kilometers
    G_automobile[u][v]['length'] = length
lengths = [edge[2]['length'] for edge in G_automobile.edges(data=True)]
shenzhen_road_graph_structure['distance'] = lengths
shenzhen_road_graph_structure['congestion_index'] = shenzhen_road_graph_structure['distance'] / (shenzhen_road_graph_structure['volume'] / 55)
shenzhen_road_graph_structure = shenzhen_road_graph_structure.reset_index()
G_automobile = nx.Graph()

for i, row in shenzhen_road_graph_structure.iterrows():
    node1 = (row['starts_lng'], row['starts_lat'])
    node2 = (row['ends_lng'], row['ends_lat'])
    G_automobile.add_node(node1, location=node1, name=shenzhen_road_graph_structure['starts_id'][i])
    G_automobile.add_node(node2, location=node2, name=shenzhen_road_graph_structure['ends_id'][i])
    G_automobile.add_edge(node1, node2, weight=shenzhen_road_graph_structure['congestion_index'][i])

route_node_list = list(G_automobile.nodes())
route_name_list = np.array([data['name'] for node, data in G_automobile.nodes(data=True)])
assert len(route_node_list) == len(route_name_list)  # Ensuring the node list and name list have the same length
route_options = [{'label': label, 'value': value} for label, value in zip(route_name_list, np.array(range(len(route_node_list))))]

# graph for Metro Station Route Planning
G = nx.Graph()
for i, row in edge3.iterrows():
    G.add_node(row['OStation'], location=(row['OLng'], row['OLat']))
    G.add_node(row['DStation'], location=(row['DLng'], row['DLat']))
    G.add_edge(row['OStation'], row['DStation'], weight=row['duration'])
metro_station_list = list(G.nodes())
station_options = [{'label': label, 'value': value} for label, value in zip(metro_station_list, np.array((range(len(metro_station_list)))))]

# Establishing Dash app server
app = Dash(__name__, external_stylesheets=[dbc.themes.COSMO], meta_tags=[{'content': 'width=device-width'}])  # Using external CSS styling
app.title = 'Traffic Flow data Tool'
app.layout = html.Div([
    html.H1('Tool for planning and visualising traffic movements', className='Header', style={'textAlign': 'center', 'fontFamily': 'Georgia', 'fontSize': '50px'}),
    dbc.Tabs([
        dbc.Tab([
            html.Ul([
                html.Div([
                    html.H3('Dear Users, thank you for using Tool for Planning and Visualising Traffic Movements!'),
                    html.H3('Please select the tabs above to corresponding functionality section, details are listed below:'),
                    dbc.Row([
                        dbc.Col(
                            [
                                html.H3('Taxi Operation Analysis'),
                                html.Ul([
                                    html.Li('Order and Duration Overview'),
                                    html.Li('Pick-up and Drop-off Block'),
                                    html.Li('Speed Overview of a Single Taxi'),
                                    html.Li('Trace of a Single Taxi'),
                                    html.Li('Orders and Duration of a Single Taxi'),
                                ], className='unordered-list')
                            ]
                        ),
                        dbc.Col(
                            [html.H3('Smart Card Analysis'),
                             html.Ul([
                                 html.Li('Swiping Overview'),
                                 html.Li('Operation Status of a Single Station'),
                                 html.Li('Peak Estimation'),
                             ], className='unordered-list')]
                        ),
                        dbc.Col(
                            [html.H3('Point of Interest'),
                             html.Ul([
                                 html.Li('Time-variant Native Clustering'),
                                 html.Li('KMeans Clustering'),
                             ], className='unordered-list')]
                        ),
                        dbc.Col(
                            [html.H3('Route Planning'),
                             html.Ul([
                                 html.Li('Road Vehicle Route Planning'),
                                 html.Li('Metro Route Planning'),
                             ], className='unordered-list')]
                        ),
                    ]),
                    html.H3('Note: Figures in the system support the following interactions: '),
                    html.Ul([
                        html.Li('Zooming — Scrolling Mouse Wheel'),
                        html.Li('Box selecting — Left-holding and dragging Mouse'),
                        html.Li('Panning — Left-holding and dragging Mouse'),
                        html.Li('Tilting — Map only, Right-holding and dragging Mouse'),
                        html.Li('Re-formatting — Double-left clicking Mouse')
                    ], className='unordered-list'),
                    html.H3('You can change interacting method by clicking the icon on the upper-right corner of each figure.')
                ], className='twelve columns pretty_container')
            ]),
        ], label='User Guide'),
        dbc.Tab([
            html.Div([
                html.Div([
                    dcc.Graph(figure=order_overview)
                ], className='six columns pretty_container', id='order-overview'),
                html.Div([
                    dcc.Graph(figure=duration_overview)
                ], className='six columns pretty_container', id='duration-overview'),
                html.Div([
                    dcc.Graph(figure=pickup_block)
                ], className='six columns pretty_container', id='pickup-overview'),
                html.Div([
                    dcc.Graph(figure=dropoff_block)
                ], className='six columns pretty_container', id='dropoff-overview'),
                html.Div([
                    html.Div([
                        html.Div([
                            html.H2('Select a taxi and your preferred trace type', style={'textAlign': 'center'}),
                            html.Span(['Note: ']),
                            html.Span('yellow ', style={'color': '#EFF82FFF', 'textShadow': '1px 1px 1px darkgray'}),
                            html.Span('refers to the occupied taxis, while '),
                            html.Span('blue ', style={'color': '#110C86FF', 'textShadow': '1px 1px 1px gray '}),
                            html.Span('refers to the available ones.')
                        ], style={'textAlign': 'center', 'color': '#000000', 'fontFamily': 'Georgia', 'fontWeight': 'bold'}),
                        dbc.Row([
                            dbc.Col([
                                html.H4('Taxi Number'),
                                dcc.Dropdown(
                                    options=[{'label': num, 'value': num}
                                             for num in np.array(list(map(str, taxi_data['VehicleNum'].unique().tolist())))],
                                    value='22223',
                                    id='select_vehicle',
                                )
                            ]),
                            dbc.Col([
                                html.H4('Preferred Trace Type'),
                                dcc.Dropdown(
                                    options=[
                                        {'label': 'All traces', 'value': 1},
                                        {'label': 'Only Occupied traces', 'value': 0}
                                    ],
                                    value=1,
                                    id='select_trace_type',
                                )
                            ])
                        ])
                    ], className='twelve columns pretty_container')
                ]),
            ], style={'margin': '0px', 'padding': '0px'}),
            dbc.Row([
                html.Div([
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(id='speed-graph')
                        )
                    ], className='six columns pretty_container', id='taxi-speed-overview'),
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(id='map-graph')
                        )
                    ], className='six columns pretty_container', id='taxi-map-display')
                ])
            ]),
            dbc.Row([
                html.Div([
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(id='order-graph')
                        )
                    ], className='six columns pretty_container', id='taxi-order-overview'),
                    html.Div([
                        dcc.Loading(
                            dcc.Graph(id='duration-graph')
                        )
                    ], className='six columns pretty_container', id='order-duration-overview')
                ]),

            ]),
            html.Div([
                html.H4('Select SVD Pattern'),
                dcc.Dropdown(
                    options=[
                        {'label': 'Pattern 1', 'value': 0},
                        {'label': 'Pattern 2', 'value': 1},
                        {'label': 'Pattern 3', 'value': 2}
                    ],
                    value=0,
                    id='select-SVD-type',
                )
            ], className='twelve columns pretty_container'),
            html.Div([
                dbc.Row([
                    html.Div([
                        html.Div([
                            dcc.Loading(
                                dcc.Graph(id='fig-SVD-temporal')
                            )
                        ], className='six columns pretty_container', id='SVD-temporal'),
                        html.Div([
                            dcc.Loading(
                                dcc.Graph(id='fig-SVD-spatial')
                            )
                        ], className='six columns pretty_container', id='SVD-spatial')
                    ])
                ]),
            ]),
        ], label='Taxi Operation Analysis'),

        dbc.Tab([
            html.Div([
                dcc.Loading(
                    dcc.Graph(id='metro-flow-overview-graph', figure=fig_metro_flow_by_station)
                )
            ], className='twelve columns pretty_container', id='metro-flow-overview'),
            html.Div([
                dcc.Loading(
                    dcc.Graph(id='metro-stations-map', figure=fig_map_metro_stations)
                )
            ], className='six columns pretty_container'),
            html.Div([
                dcc.Loading(
                    dcc.Graph(id='metro-flow-graph-by-region', figure=fig_metro_flow_by_region)
                )
            ], className='six columns pretty_container', id='metro-flow-by-region'),
            html.Div([
                html.H4('Station Name'),
                dcc.Dropdown(
                    options=[{'label': station, 'value': station}
                             for station in np.array(list(map(str, smart_cardID_data['Station'].unique().tolist())))],
                    value='益田',
                    id='station-dropdown'
                ),
                html.Div([
                    dcc.Loading(
                        dcc.Graph(id='metro-flow-graph')
                    )
                ])
            ], className='six columns pretty_container', id='metro-flow-specific'),
            html.Div([
                dcc.Loading(
                    dcc.Graph(id='metro-flow-peak-morning-graph', figure=metro_flow_peak_morning)
                ),
                dcc.Loading(
                    dcc.Graph(id='metro-flow-peak-evening-graph', figure=metro_flow_peak_evening)
                )
            ], className='six columns pretty_container', id='metro-flow-peak'),

        ], label='Metro Smart Card Analysis'),
        dbc.Tab([
            html.Div([
                html.Div(
                    html.H2('Point of Interest Clustering', style={'textAlign': 'center'})
                ),
                dbc.Row([
                    dbc.Col(dcc.Dropdown(
                        options=[
                            {'label': 'Native Clustering', 'value': 0},
                            {'label': 'K-means Clustering', 'value': 1}
                        ],
                        value='0',
                        id='select_clustering_type',

                    )),
                    dbc.Col(dcc.Slider(
                        min=1,
                        max=10,
                        step=1,
                        marks={i: str(i) for i in range(1, 11)},
                        value=5,
                        updatemode="drag",
                        id='select_K_means_level',

                    ))
                ]
                )
            ]),
            html.Div([
                dcc.Loading(
                    dcc.Graph(id='POI_clustering')
                )
            ], className='twelve columns pretty_container')
        ], label='Points of Interest'),
        dcc.Tab([
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            options=route_options,
                            value=0,
                            id='select-route-origin',
                        )
                    ]),
                    dbc.Col([
                        dcc.Dropdown(
                            options=route_options,
                            value=0,
                            id='select-route-destination',
                        )
                    ])
                ]),
                dbc.Tabs([
                    dbc.Tab(
                        html.Div([
                            dcc.Loading(
                                dcc.Graph(id='fig-route-optimization')
                            )
                        ])
                        , label='Abstract Network'),
                    dbc.Tab(
                        html.Div([
                            dcc.Loading(
                                dcc.Graph(id='fig-route-optimization-map')
                            )
                        ])
                        , label='Real Map')
                ])
            ], id='route-optimization', className='twelve columns pretty_container'),
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            # --------------------------------------------------------------------------
            html.Div([
                dbc.Row([
                    dbc.Col([
                        dcc.Dropdown(
                            options=station_options,
                            value=0,
                            id='select-metro-origin',
                        )
                    ]),
                    dbc.Col([
                        dcc.Dropdown(
                            options=station_options,
                            value=0,
                            id='select-metro-destination',
                        )
                    ])
                ]),
                html.Div([
                    html.H4(
                        id='metro-route-optimization-result'
                    ),
                    dcc.Loading(
                        dcc.Graph(id='fig-metro-optimization')
                    )

                ])
            ], id='metro-optimization', className='twelve columns pretty_container')
        ], label='Route Planning')
    ]),
], id='Main_page')


# Callback Functions
# updating figures about single taxi operation
@app.callback(
    [
        Output('speed-graph', 'figure'),
        Output('map-graph', 'figure'),
        Output('order-graph', 'figure'),
        Output('duration-graph', 'figure')
    ],
    [
        Input('select_vehicle', 'value'),
        Input('select_trace_type', 'value')
    ]
)
def update_taxi_figures(vehicleNum, mapType):
    if not vehicleNum:
        raise PreventUpdate
    # Data preprocessing
    filtered_taxi = taxi_data[taxi_data.VehicleNum == int(vehicleNum)].sort_values('Stime')
    taxi_OD_data = taxi_OD_data_overview[taxi_OD_data_overview.VehicleNum == int(vehicleNum)].sort_values('Stime')

    fig_speed = go.Figure(
        data=[
            go.Scatter(
                mode='markers+lines',
                x=filtered_taxi['Stime'],
                y=filtered_taxi['Speed'],
                marker=dict(size=2, color=('#110C86' if taxi_OD_data.empty else filtered_taxi['OpenStatus'])),
                line=dict(width=1, color='#A6824E', shape='spline'),
                hovertemplate='<b>%{y}</b> <i>Km/h</i><extra></extra>' +
                              '<br>at <b>%{x}</b>'
            )
        ],
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(l=20, r=20, t=50, b=40),
            title='The Speed Overview of Taxi ' + vehicleNum,
            hoverlabel=dict(
                font_size=16,
                font_family="Rockwell"
            ),
        )
    )
    fig_speed.update_xaxes(automargin=True, showgrid=False)
    fig_speed.update_layout(
        xaxis=dict(
            title='Time',
            rangeslider=dict(
                visible=True
            ),
            type='category'
        ),
        yaxis=dict(
            title='Speed'
        ),
        height=500,

        autosize=True,
    )
    # Considering possibilities that taxi has no orders
    if taxi_OD_data.empty:
        fig_bar = {
            'data': [
                {
                    'type': 'indicator',
                    'title': 'Sorry, the selected object has no records of orders.<br>Please select another object.',
                    'number': {'font': {'color': '#263238'}}
                }
            ],
            'layout': {
                'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10}
            },
        }
        fig_duration = {
            'data': [
                {
                    'type': 'indicator',
                    'title': 'Sorry, the selected object has no records of orders.<br>Please select another object.',
                    'number': {'font': {'color': '#263238'}}
                }
            ],
            'layout': {
                'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10}
            },
        }
    else:
        taxi_OD_data['Hour'] = taxi_OD_data.apply(lambda r: r['Stime'][:2], axis=1).astype(int)
        hour_count_OD = taxi_OD_data.groupby('Hour')['VehicleNum'].count()
        hour_count_OD = hour_count_OD.rename('count').reset_index()
        fig_bar = go.Figure(
            data=[go.Bar(
                x=hour_count_OD['Hour'],
                y=hour_count_OD['count']
            )],
            layout=go.Layout(
                autosize=True,
                margin=go.layout.Margin(l=20, r=20, t=50, b=40),
                title='The Order Overview of Taxi ' + vehicleNum,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=16,
                    font_family="Rockwell"
                )
            )
        )
        fig_bar.update_xaxes(tick0=0, dtick=1)
        fig_bar.update_yaxes(tick0=0, dtick=1)
        fig_bar.update_layout(
            xaxis=dict(title='Hour'),
            yaxis=dict(title='Number of Orders'),

            autosize=True
        )

        fig_duration = go.Figure(
            data=[
                go.Box(
                    x=taxi_OD_data['Hour'],
                    y=taxi_OD_data['duration'] / 60,
                    boxmean='sd',  # represent mean and standard deviation
                    quartilemethod="exclusive"
                )
            ],
            layout=go.Layout(
                autosize=True,
                margin=go.layout.Margin(l=20, r=20, t=50, b=40),
                title='The Duration Overview of Taxi ' + vehicleNum,
                hoverlabel=dict(
                    bgcolor="white",
                    font_size=16,
                    font_family="Rockwell"
                )
            )
        )
        fig_duration.update_xaxes(tick0=0, dtick=1)
        fig_duration.update_layout(
            xaxis=dict(
                title='Hour',
                type='category'
            ),
            yaxis=dict(
                title='Duration (minute)'
            ),

        )

    if mapType == 1:
        fig_map = go.Figure(
            data=[
                go.Scattermapbox(
                    mode='markers+lines',
                    lat=filtered_taxi['Lat'],
                    lon=filtered_taxi['Lng'],
                    text=filtered_taxi['Stime'],
                    marker=dict(size=4,
                                color=('#110C86' if taxi_OD_data.empty else filtered_taxi['OpenStatus'])
                                ),
                    line=dict(width=2, color='#A6824E')
                )
            ],
            layout=go.Layout(
                title='Traces of Taxi ' + vehicleNum,
                autosize=True,
                margin=go.layout.Margin(l=10, r=30, t=50, b=30),
                hoverlabel=dict(
                    font_size=16,
                    font_family="Rockwell"
                )
            )
        )
    else:
        if taxi_OD_data.empty:
            fig_map = {
                'data': [
                    {
                        'type': 'indicator',
                        'title': 'Sorry, the selected object has no records of passenger-occupied status.<br>Please change your <i><b>preferred trace type</b></i> or select another object.',
                        'number': {'font': {'color': '#263238'}}
                    }
                ],
                'layout': {
                    'margin': {'l': 10, 'r': 10, 't': 10, 'b': 10}
                },
            }
        else:
            fig_map = go.Figure(
                data=[go.Scattermapbox(
                    mode='markers+lines',
                    lat=taxi_OD_data['SLat'].round(3),
                    lon=taxi_OD_data['SLng'].round(3),
                    marker=dict(size=8, color='yellow'),
                )
                ],
                layout=go.Layout(
                    title='Passenger Path of Taxi ' + vehicleNum,
                    autosize=True,
                    margin=go.layout.Margin(l=10, r=30, t=50, b=30),
                    hoverlabel=dict(
                        font_size=16,
                        font_family="Rockwell"
                    )
                )
            )

    fig_map.update_layout(
        mapbox={
            'center': {'lon': lonInitial, 'lat': latInitial},
            'style': 'open-street-map',
            'zoom': zoom},
        showlegend=False,
        height=500,
    )

    return fig_speed, fig_map, fig_bar, fig_duration


# updating firgures about SVD operation
@app.callback(
    [
        Output('fig-SVD-temporal', 'figure'),
        Output('fig-SVD-spatial', 'figure')
    ],
    Input('select-SVD-type', 'value')
)
def update_SVD(SVD_type):
    U_pattern = pd.DataFrame([U[:, SVD_type], Gridmatrix.index]).T  # Calculating matrix according to the SVD type selected by users
    U_pattern.columns = ['U', 'Hour']
    U_pattern['Hour'] = U_pattern['Hour'].astype(int).astype(str)
    V_pattern = pd.DataFrame([VT.T[:, SVD_type], Gridmatrix.columns]).T
    V_pattern.columns = ['V', 'grid_id']
    V_pattern['LngCol'] = V_pattern['grid_id'].apply(lambda r: r.split('|')[0]).astype(int)
    V_pattern['LatCol'] = V_pattern['grid_id'].apply(lambda r: r.split('|')[1]).astype(int)
    V_pattern['V'] = V_pattern['V'].astype(float)
    grid_toplot = pd.merge(grid, V_pattern, on=['LngCol', 'LatCol'])
    fig_SVD_temporal = go.Figure(
        data=[
            go.Bar(x=U_pattern['Hour'], y=U_pattern['U']),
            go.Scatter(x=U_pattern['Hour'], y=U_pattern['U'], mode='markers+lines')
        ],
        layout=go.Layout(
            xaxis=dict(title='Time'),
            yaxis=dict(title='U'.join(str(i))),
            title='SVD Type ' + str(SVD_type),
            autosize=True,
            showlegend=False,
            margin=go.layout.Margin(l=10, r=30, t=50, b=30),
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            )
        )
    )
    fig_SVD_spatial = px.choropleth_mapbox(grid_toplot, geojson=grid_toplot.geometry, locations=grid_toplot.index, color='V')
    fig_SVD_spatial.update_layout(
        mapbox={
            'center': {'lon': lonInitial, 'lat': latInitial},
            'style': 'open-street-map',
            'zoom': 9},
        autosize=True,
        margin=go.layout.Margin(l=10, r=30, t=50, b=30),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        )
    )
    return fig_SVD_temporal, fig_SVD_spatial


# updating figuers about traffic flow in single metro station
@app.callback(
    Output('metro-flow-graph', 'figure'),
    Input('station-dropdown', 'value')
)
def update_metro_figures(stationName):
    if not stationName:
        raise PreventUpdate
    filter_df = smart_cardID_data[smart_cardID_data['Station'] == stationName]
    filter_df_in = filter_df[filter_df['TransactionType'] == 'Swipe-In']
    filter_df_out = filter_df[filter_df['TransactionType'] == 'Swipe-Out']
    tmp_in = filter_df_in.groupby(by=['Time', 'Station'])['SmartCardID'].count().rename('count').reset_index()
    tmp_out = filter_df_out.groupby(by=['Time', 'Station'])['SmartCardID'].count().rename('count').reset_index()
    fig_metro_flow = go.Figure(
        data=[
            go.Scatter(x=tmp_in['Time'], y=tmp_in['count'], name='Swipe-In', line=dict(shape='spline', color='royalblue')),
            go.Scatter(x=tmp_out['Time'], y=tmp_out['count'], name='Swipe-Out', line=dict(shape='spline', color='firebrick'))
        ],
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(l=20, r=20, t=50, b=40),
            title='The Traffic Flow of Station ' + stationName,
            hoverlabel=dict(
                font_size=16,
                font_family="Rockwell"
            ),
            xaxis=dict(title='Hour', type='category', categoryorder='category ascending', automargin=True, showgrid=False),
            yaxis=dict(title='Number of Swipes'),

            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.8,
                bgcolor='rgba(0,0,0,0)'
            ),
        )
    )

    return fig_metro_flow


# updating tbe figure of POI discovery
@app.callback(
    Output('POI_clustering', 'figure'),
    [
        Input('select_clustering_type', 'value'),
        Input('select_K_means_level', 'value')
    ]
)
def update_clustering(cluster_type, cluster_level):
    global k_value

    # KMeans clustering
    if cluster_type == 1:
        k_value = cluster_level
        temp1 = KMeans(n_clusters=cluster_level, random_state=9, n_init=cluster_level)
        temp1 = temp1.fit(data_to_cluster_zip)
        output1 = temp1.predict(data_to_cluster_zip)
        data_to_cluster['type'] = output1
        fig_cluster = px.scatter_mapbox(data_to_cluster, lon='Lng', lat='Lat', size=data_to_cluster['count'], color=data_to_cluster['type'])
    else:
        fig_cluster = px.scatter_mapbox(CDR_data, lon='Lng', lat='Lat', animation_frame='Hour', size=CDR_data['count'] * 2, color=CDR_data['count'])
        fig_cluster.update_layout(
            transition_duration=500,
            updatemenus=[
                dict(type='buttons', showactive=True, buttons=[dict(label='Play', method='animate', args=[None, {'frame': {'duration': 1000, 'redraw': True}, 'transition': {'duration': 1000, 'easing': 'quadratic-in-out'}}]),
                                                               dict(label='Pause', method='animate', args=[[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}])])
            ]
        )
    fig_cluster.update_layout(
        mapbox={
            'center': {'lon': lonInitial, 'lat': latInitial},
            'style': 'open-street-map',
            'zoom': zoom
        },
        title='Cluster',
        autosize=True,
        showlegend=False,
        margin=go.layout.Margin(l=10, r=30, t=50, b=30),
        height=700,
        hoverlabel=dict(
            font_size=16,
            font_family="Rockwell"
        ),
    )
    return fig_cluster


# updating the figures of road vehicle route optimization
@app.callback(
    [
        Output('fig-route-optimization', 'figure'),
        Output('fig-route-optimization-map', 'figure')
    ],
    [
        Input('select-route-origin', 'value'),
        Input('select-route-destination', 'value')
    ]
)
def automobile_route_optimization(origin, destination):
    result = nx.shortest_path(G_automobile, source=route_node_list[origin], target=route_node_list[destination], weight='weight')
    result_names = np.array([G_automobile.nodes[node_id]['name'] for node_id in result])
    result_lats = np.array([G_automobile.nodes[node_id]['location'][0] for node_id in result])
    result_lons = np.array([G_automobile.nodes[node_id]['location'][1] for node_id in result])
    fig_route_optimization = go.Figure(
        data=[
            go.Scatter(
                x=shenzhen_road['Lng'],
                y=shenzhen_road['Lat'],
                mode='markers+lines',
                name='Raw Route',
                marker=dict(
                    color='#223242',
                    line=dict(color='white', width=4),
                    size=6),
                hovertemplate='%{text}',
                text=[f'{intersection_name}' for intersection_name in shenzhen_road['Intersection'] + ' with ' + shenzhen_road['Road']],
                textposition='bottom center'
            ),
            go.Scatter(
                x=result_lons,
                y=result_lats,
                mode='markers+lines',
                name='Optimized Route',
                marker=dict(
                    color='firebrick',
                    line=dict(color='firebrick', width=2),
                    size=6,
                    symbol='arrow',
                    angleref='previous',
                    standoff=6
                ),
                hovertemplate='%{text}',
                text=[f'Checkpoint: {intersection_name}' for intersection_name in result_names]
            )
        ],
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(l=10, r=30, t=50, b=30),
            height=800,
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
            xaxis=dict(
                showline=True,
                showgrid=False,
                showticklabels=True,
            ),
            yaxis=dict(showgrid=False),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0)'
            ),

            plot_bgcolor='rgb(243, 243, 243)'
        ),
    )
    fig_route_optimization_map = go.Figure(
        data=[
            go.Scattermapbox(
                lon=shenzhen_road['Lng'],
                lat=shenzhen_road['Lat'],
                mode='lines+markers',
                text=[f'{intersection_name}' for intersection_name in shenzhen_road['Intersection'] + ' with ' + shenzhen_road['Road']],
                name='Raw Route',
                marker=dict(
                    color='#223242',
                    size=4
                ),
            ),
            go.Scattermapbox(
                lat=result_lats,
                lon=result_lons,
                mode='lines+markers',
                name='Optimized Route',
                marker=dict(
                    color='red',
                    size=4
                ),
                hovertemplate='%{text}',
                text=[f'Checkpoint: {intersection_name}' for intersection_name in result_names]
            )
        ],
        layout=go.Layout(
            autosize=True,
            margin=go.layout.Margin(l=10, r=30, t=50, b=30),
            height=800,
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,
                font_family="Rockwell"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0)'
            ),
            mapbox={
                'center': {'lon': 113.9539, 'lat': 22.54346},
                'style': 'open-street-map',
                'zoom': zoom + 2}
        )
    )

    return fig_route_optimization, fig_route_optimization_map


# updating the figure of metro network route optimization
@app.callback(
    [
        Output('fig-metro-optimization', 'figure'),
        Output('metro-route-optimization-result', 'children')
    ],
    [
        Input('select-metro-origin', 'value'),
        Input('select-metro-destination', 'value')
    ]
)
def metro_route_optimization(origin, destination):
    result = np.array(nx.shortest_path(G, source=metro_station_list[origin], target=metro_station_list[destination], weight='weight'))
    '''
    result_locations = [G.nodes[node_id] for node_id in result]
    result_lons = [item['location'][0] for item in result_locations]
    result_lats = [item['location'][1] for item in result_locations]
    '''
    result_locations = np.array([G.nodes[node_id]['location'] for node_id in result])
    result_lons = result_locations[:, 0]
    result_lats = result_locations[:, 1]
    fig_metro_optimization = px.line(
        metro_network,
        x='Lngs', y='Lats',
        color='Lines',
        line_shape='spline',
        labels={'color': 'Metro Line'},
    )
    fig_metro_optimization.update_traces(
        line=dict(width=8)
    )
    fig_metro_optimization.add_trace(
        go.Scatter(
            x=metro_network['Lngs'], y=metro_network['Lats'], mode='markers+text',
            hovertext=metro_network['Stops'],
            name='Metro Station',
            line=dict(width=4, shape='spline'),
            text=metro_network['Names'],
            textposition='bottom center',
            marker=dict(
                color='#FFFFFF',
                size=8,
                line=dict(
                    color='#000232',
                    width=2
                )
            ),
        )
    )
    fig_metro_optimization.add_trace(
        go.Scatter(
            x=result_lons,
            y=result_lats,
            mode='markers+lines',
            name='Optimized Route',
            hovertext=result,
            marker=dict(
                color='firebrick',
                line=dict(color='white', width=4),
                size=5,
                symbol='arrow',
                angleref='previous',
                standoff=8
            ),
            line=dict(width=4, color='firebrick', shape='spline')

        )
    )
    fig_metro_optimization.update_layout(
        height=1000,
        autosize=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(0,0,0,0)'
        ),
        margin=go.layout.Margin(l=10, r=10, t=40, b=20),
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )
    fig_metro_optimization.update_xaxes(showticklabels=False, title=None)
    fig_metro_optimization.update_yaxes(showticklabels=False, title=None)
    return fig_metro_optimization, process_list(result)


# adding arrows between elements in list
def process_list(lst):
    result_str = lst[0]
    for i in range(1, len(lst)):
        if lst[i] != lst[i - 1]:
            result_str += ' → ' + lst[i]
        else:
            result_str += lst[i]
    return result_str


# setting up server
# Enable debug mode allowing developer to inspect the performance and possible bugs
if __name__ == '__main__':
    app.run_server(debug=True)
