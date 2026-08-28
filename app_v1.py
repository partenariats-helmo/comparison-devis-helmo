import json
import os
import re
import time
import pandas as pd
import streamlit as st
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import errors, types
from supabase import create_client, Client

# --- CONFIGURATION STREAMLIT & STYLES ---
st.set_page_config(page_title="HELMO — Analyse & Comparaison", layout="wide")

# Correctif CSS : évite le tronquage des premiers caractères et gère les retours à la ligne
st.markdown("""
    <style>
    div[data-testid="stDataFrame"] table {
        white-space: normal !important;
    }
    div[data-testid="stDataFrame"] td {
        word-break: break-word !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("HELMO — Analyse & Comparaison de Devis")

TEXTE_EXPERT = "Prestation en étude par un expert HELMO"
TEL_EXPERT = "+33 9 78 45 08 04"
MODEL_NAME = "gemini-3.6-flash"

# --- CLEFS ET APIS ---
try:    
    GEMINI_KEY = st.secrets["GEMINI_API_KEY"]    
    SUPABASE_URL = st.secrets["SUPABASE_URL"]    
    SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
        
except Exception:    
    GEMINI_KEY = os.getenv("GEMINI_API_KEY")    
    SUPABASE_URL = os.getenv("SUPABASE_URL")    
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")    


if not GEMINI_KEY or not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Clés d'API manquantes. Assure-toi de configurer .streamlit/secrets.toml ou tes variables d'environnement.")
    st.stop()

client = genai.Client(api_key=GEMINI_KEY)
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- SCHÉMAS PYDANTIC ---
class LineItem(BaseModel):
    famille_raw: Optional[str] = Field(description="Catégorie ou famille mentionnée sur le devis")
    designation_raw: str = Field(description="Description détaillée de la prestation client")
    quantite: float = Field(default=1.0, description="Quantité réelle")
    unite: str = Field(default="u", description="Unité de mesure (m², ml, u, forfait, etc.)")
    prix_unitaire_ht: float = Field(default=0.0, description="Prix unitaire HT")
    total_ht: float = Field(default=0.0, description="Total HT de la ligne")

class DevisParsed(BaseModel):
    surface_globle_m2: Optional[float] = Field(description="Surface globale du logement en m²")
    type_logement: Optional[str] = Field(description="Type de logement (Appartement, Maison, etc.)")
    lignes: List[LineItem] = Field(description="Liste des prestations extraites du devis")

# --- FONCTIONS UTILITAIRES ---
def extraire_prix_numerique(item_dict: dict) -> float:
    if not item_dict or not isinstance(item_dict, dict):
        return 0.0
    
    cles_prix = ["prix_unitaire_ht", "prix_ht", "pu_ht", "prix_unitaire", "tarif_ht", "pu", "prix"]
    valeur_prix = None
    
    for cle in cles_prix:
        if cle in item_dict and item_dict[cle] is not None:
            valeur_prix = item_dict[cle]
            break
            
    if valeur_prix is None:
        return 0.0
        
    if isinstance(valeur_prix, (int, float)):
        return float(valeur_prix)
        
    chaine_nettoyee = re.sub(r"[^\d.,]", "", str(valeur_prix)).replace(",", ".")
    try:
        return float(chaine_nettoyee)
    except ValueError:
        return 0.0

def determiner_unite_affichage(unite_str: str) -> str:
    """Détermine précisément l'unité pour éviter le bug où 'unité' contient la lettre 'm'."""
    unite_clean = str(unite_str).lower().strip()
    if "m²" in unite_clean or "m2" in unite_clean:
        return "m²"
    elif "ml" in unite_clean or "mètre linéaire" in unite_clean:
        return "ml"
    elif "forfait" in unite_clean:
        return "forfait"
    else:
        return "u"

def colorier_tableau(row):
    styles = [''] * len(row)
    if row["_is_expert"]:
        return ['background-color: #fffde7'] * len(row)
    if row["Écart (€)"] > 0:
        return ['background-color: #ffebee'] * len(row)
    return styles

def appeler_gemini_avec_retry(contents, config=None, max_retries=5):
    """Gère le rate limit 429 avec pause automatique."""
    for tentative in range(max_retries):
        try:
            return client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=config
            )
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                temps_attente = 35 + (tentative * 10)
                st.warning(f"⚠️ Quota Gemini atteint. Pause automatique de {temps_attente} secondes...")
                time.sleep(temps_attente)
            else:
                raise e
        except Exception as e:
            if tentative == max_retries - 1:
                raise e
            time.sleep(3)
    raise RuntimeError("Nombre maximal de tentatives atteint.")

# --- ANALYSE & MATCHING ---
def analyser_devis_bytes(file_bytes: bytes, mime_type: str, progress_bar) -> DevisParsed:
    progress_bar.progress(20, text=f"Étape 1/2 : Analyse du devis ({MODEL_NAME})...")
    prompt = """
    Extrais toutes les prestations, montants et détails structurés de ce devis de BTP.
    Analyse très attentivement le document pour identifier ou additionner la surface globale du logement en m².
    Si la quantité sur la ligne indique "1 u" mais que la description contient une surface explicite, privilégie la quantité réelle.
    Renseigne également le type de logement.
    """
    
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=DevisParsed,
        temperature=0.1,
    )
    
    response = appeler_gemini_avec_retry(
        contents=[types.Part.from_bytes(data=file_bytes, mime_type=mime_type), prompt],
        config=config
    )
    progress_bar.progress(100, text="Étape 1/2 : Extraction terminée !")
    return DevisParsed.model_validate_json(response.text)

def matcher_un_lot_de_lignes(lignes_batch, catalogue_helmo, surface_m2: Optional[float]):
    PROMPT_MATCHING_BATCH = """
    Tu es un expert en chiffrage BTP HELMO.
    Surface globale estimée : {surface_m2} m²

    LIGNES CLIENT À MATCHER :
    {lignes_json}

    CATALOGUE HELMO DISPONIBLE :
    {candidates_json}

    Consignes de matching :
    1. Pour chaque ligne client, sélectionne le ou les articles HELMO correspondants.
    2. Pour les PRESTATIONS COMPOSÉES, associe TOUS les articles correspondants sous "matches".
    3. Si aucun équivalent exact, renvoie "matches": [].

    Format JSON attendu :
    [
      {{
        "index_ligne": 0,
        "matches": [
          {{
            "id_tarif_selected": <id_helmo>,
            "qte_helmo": <quantite_ou_null>,
            "confidence_score": 85,
            "raison": "<explication_courte>"
          }}
        ]
      }}
    ]
    """
    candidates_light = [
        {
            "id": c.get("id_tarif"),
            "prod": c.get("produit") or c.get("designation"),
            "unit": c.get("unite_tarif") or c.get("unite"),
            "pu": extraire_prix_numerique(c)
        }
        for c in catalogue_helmo
    ]

    prompt = PROMPT_MATCHING_BATCH.format(
        surface_m2=surface_m2 or "Inconnue",
        lignes_json=json.dumps(lignes_batch, ensure_ascii=False),
        candidates_json=json.dumps(candidates_light, ensure_ascii=False)
    )

    try:
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.0
        )
        response = appeler_gemini_avec_retry(contents=prompt, config=config)
        return json.loads(response.text)
    except Exception:
        return []

def matcher_toutes_les_lignes_optimise(lignes_client, catalogue_helmo, surface_m2, batch_size=10):
    lignes_formatted = [
        {
            "index_ligne": idx,
            "famille": l.famille_raw or "",
            "designation": l.designation_raw,
            "quantite": l.quantite,
            "unite": l.unite,
            "total_ht": l.total_ht
        }
        for idx, l in enumerate(lignes_client)
    ]

    tous_les_matches = []
    total_lots = (len(lignes_formatted) + batch_size - 1) // batch_size
    progress_bar = st.progress(0, text="Étape 2/2 : Matching HELMO par lots...")

    for i in range(0, len(lignes_formatted), batch_size):
        batch = lignes_formatted[i:i + batch_size]
        res_batch = matcher_un_lot_de_lignes(batch, catalogue_helmo, surface_m2)
        if isinstance(res_batch, list):
            tous_les_matches.extend(res_batch)
        
        current_batch = (i // batch_size) + 1
        progress_bar.progress(current_batch / total_lots, text=f"Étape 2/2 : Matching lot {current_batch}/{total_lots}...")
        time.sleep(2)

    progress_bar.empty()
    return tous_les_matches

# --- INTERFACE STREAMLIT ---
uploaded_file = st.file_uploader("Téléverse un devis (PDF, PNG, JPG)", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    if st.button("Lancer l'Analyse et le Matching", type="primary"):
        
        progress_bar_etape1 = st.progress(0, text="Analyse du document...")
        bytes_data = uploaded_file.getvalue()
        mime_type = uploaded_file.type
        resultat = analyser_devis_bytes(bytes_data, mime_type, progress_bar_etape1)
        progress_bar_etape1.empty()

        st.success("Extraction du devis réussie !")
        
        col1, col2 = st.columns(2)
        surface_val = resultat.surface_globle_m2 if resultat.surface_globle_m2 else 0.0
        logement_val = resultat.type_logement if resultat.type_logement else "Appartement"

        surface_globle = col1.number_input("Surface globale estimée (m²)", value=float(surface_val), step=1.0)
        type_logement = col2.text_input("Type de logement", value=str(logement_val))

        res = supabase.table("tarifs_helmo").select("*").execute()
        catalogue_helmo = res.data or []

        matches = matcher_toutes_les_lignes_optimise(
            resultat.lignes, 
            catalogue_helmo, 
            surface_globle, 
            batch_size=10
        )
        matches_by_idx = {m.get("index_ligne"): m for m in matches if isinstance(m, dict)}

        resultats_combines = []
        for idx, ligne in enumerate(resultat.lignes):
            match_entry = matches_by_idx.get(idx, {})
            matches_list = match_entry.get("matches", [])
            valid_matches = [m for m in matches_list if m.get("id_tarif_selected") is not None]

            # Vérification si un des matchs renvoie un prix nul (ex: ID 606 à 0,00 €)
            pu_valide = True
            if valid_matches:
                for m in valid_matches:
                    h_item = next((c for c in catalogue_helmo if str(c.get("id_tarif")) == str(m.get("id_tarif_selected"))), None)
                    if not h_item or extraire_prix_numerique(h_item) <= 0:
                        pu_valide = False
                        break

            # CAS 1 : Aucun match OU prix catalogue égal à 0 € -> Bascule en étude expert
            if not valid_matches or not pu_valide:
                qte_eff = ligne.quantite if ligne.quantite > 0 else 1.0
                pu_client = ligne.total_ht / qte_eff if ligne.total_ht > 0 else 0.0

                resultats_combines.append({
                    "Famille Client": ligne.famille_raw or "",
                    "Désignation Client": ligne.designation_raw,
                    "Qté": ligne.quantite,
                    "Unité": ligne.unite,
                    "Total HT Client (€)": ligne.total_ht,
                    "ID HELMO": "—",
                    "Produit HELMO": TEXTE_EXPERT,
                    "Prix U. HELMO (€)": f"{pu_client:,.2f} €".replace(",", " ").replace(".", ","),
                    "Total HT HELMO (€)": ligne.total_ht,
                    "Écart (€)": 0.0,
                    "Raison": f"Prestation non chiffrée ou sur-mesure. Évaluation en cours par nos experts ({TEL_EXPERT}).",
                    "_is_expert": True
                })
            # CAS 2 : Match valide avec tarif HELMO > 0 €
            else:
                ids_helmo = []
                produits_helmo = []
                sous_pu_helmo = []
                total_ht_helmo_cumule = 0.0
                raisons_details = []

                for m in valid_matches:
                    selected_id = m.get("id_tarif_selected")
                    qte_override = m.get("qte_helmo")
                    
                    helmo_item = next((c for c in catalogue_helmo if str(c.get("id_tarif")) == str(selected_id)), None)
                    if helmo_item:
                        ids_helmo.append(str(selected_id))
                        nom_prod = helmo_item.get("produit") or helmo_item.get("designation") or "Article HELMO"
                        produits_helmo.append(nom_prod)
                        
                        pu_sub = extraire_prix_numerique(helmo_item)
                        qte_sub = qte_override if qte_override is not None else (ligne.quantite if ligne.quantite > 0 else 1.0)
                        
                        # Détection d'unité fiabilisée (évite 'Unité' -> 'm²')
                        raw_unit = str(helmo_item.get("unite_tarif") or helmo_item.get("unite") or ligne.unite)
                        unit_str = determiner_unite_affichage(raw_unit)
                        
                        sous_pu_helmo.append(f"{pu_sub:,.2f} €/{unit_str}".replace(",", " ").replace(".", ","))

                        if unit_str == "forfait":
                            total_ht_helmo_cumule += pu_sub
                        else:
                            total_ht_helmo_cumule += (pu_sub * qte_sub)
                        
                        raisons_details.append(
                            f"Match sur '{nom_prod}' ({qte_sub:.2f} {ligne.unite} × {pu_sub:,.2f} €/{unit_str} = {pu_sub * qte_sub:,.2f} € HT)"
                        )

                str_prix_u = " + ".join(sous_pu_helmo)
                ecart = total_ht_helmo_cumule - ligne.total_ht

                raison_txt = " | ".join(raisons_details).replace(",", " ").replace(".", ",")
                if ecart > 0:
                    raison_txt = f"Prix HELMO supérieur (+{ecart:.2f} €) : exigences de matériaux et pose certifiée. | {raison_txt}"

                resultats_combines.append({
                    "Famille Client": ligne.famille_raw or "",
                    "Désignation Client": ligne.designation_raw,
                    "Qté": ligne.quantite,
                    "Unité": ligne.unite,
                    "Total HT Client (€)": ligne.total_ht,
                    "ID HELMO": ", ".join(ids_helmo),
                    "Produit HELMO": " + ".join(produits_helmo),
                    "Prix U. HELMO (€)": str_prix_u,
                    "Total HT HELMO (€)": total_ht_helmo_cumule,
                    "Écart (€)": ecart,
                    "Raison": raison_txt,
                    "_is_expert": False
                })

        df_raw = pd.DataFrame(resultats_combines)

        total_client_global = df_raw["Total HT Client (€)"].sum()
        total_helmo_global = df_raw["Total HT HELMO (€)"].sum()
        ecart_global = total_helmo_global - total_client_global

        st.markdown("### Synthèse financière globale")
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Devis Client", f"{total_client_global:,.2f} €".replace(",", " ").replace(".", ","))
        
        # Total HELMO avec mention explicite sous la métrique
        m2.metric("Total Devis HELMO", f"{total_helmo_global:,.2f} €".replace(",", " ").replace(".", ","))
        m2.caption("⚠️ **À confirmer avec un expert HELMO**")

        m3.metric("Écart Total", f"{ecart_global:,.2f} €".replace(",", " ").replace(".", ","), delta=f"{ecart_global:,.2f} €", delta_color="inverse")

        col_info, col_btn = st.columns([3, 1])
        nb_etudes = len(df_raw[df_raw["_is_expert"] == True])
        with col_info:
            if nb_etudes > 0:
                st.warning(f"⚠️ **{nb_etudes} prestation(s)** sont en cours d'évaluation par nos experts (lignes surlignées en jaune).")
        with col_btn:
            st.link_button("📞 Contacter un expert HELMO", f"tel:{TEL_EXPERT.replace(' ', '')}", type="primary")

        cols_affichage = [
            "Famille Client", "Désignation Client", "Qté", "Unité", "Total HT Client (€)",
            "ID HELMO", "Produit HELMO", "Prix U. HELMO (€)", "Total HT HELMO (€)",
            "Écart (€)", "Raison"
        ]

        df_styled = df_raw.style.apply(colorier_tableau, axis=1).format({
            "Qté": "{:.2f}",
            "Total HT Client (€)": "{:,.2f} €",
            "Total HT HELMO (€)": "{:,.2f} €",
            "Écart (€)": "{:,.2f} €"
        })

        st.markdown("### Tableau comparatif détaillé")
        st.dataframe(df_styled, column_order=cols_affichage, use_container_width=True)
