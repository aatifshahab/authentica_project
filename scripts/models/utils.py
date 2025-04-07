import folium

def create_map(lat, lon, region_label, map_filename):
    m = folium.Map(location=[lat, lon], zoom_start=10)
    folium.Marker(
        location=[lat, lon],
        tooltip=f"Predicted Region: {region_label}"
    ).add_to(m)
    m.save(map_filename)
    return map_filename
