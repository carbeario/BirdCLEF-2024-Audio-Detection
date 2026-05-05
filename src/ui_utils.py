import streamlit as st


def get_bird_links(scientific_name, ebird_code=None):
    """
    Recibe el nombre científico Y el código eBird explícitamente.
    No busca nada, solo construye URLs
    """
    q_url = scientific_name.replace(" ", "+")
    q_wiki = scientific_name.replace(" ", "_")
    
    # Si nos pasan el código (ej: rewbul), lo usamos. Si no, fallback al nombre.
    ebird_slug = ebird_code if ebird_code else q_url
    
    return {
        "ebird": f"https://ebird.org/species/{ebird_slug}", # <--- URL CORRECTA
        "xeno-canto": f"https://xeno-canto.org/explore?query={q_url}",
        "wikipedia": f"https://en.wikipedia.org/wiki/{q_wiki}",
        "google_imgs": f"https://www.google.com/search?tbm=isch&q={q_url}"
    }