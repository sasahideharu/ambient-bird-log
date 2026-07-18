import streamlit as st
import pandas as pd
import os
import io
import requests
import re
from datetime import datetime
from dotenv import load_dotenv
from supabase import create_client, Client
from scipy.io import wavfile
import uuid

# --- 1. 環境変数とSupabase接続 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, "../.env"))

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "bird-wav"

if not url or not key:
    st.error("⚠️ .envファイルからSupabaseのAPIキーが見つからないぜ。")
    st.stop()

supabase: Client = create_client(url, key)

# --- 2. リモート音声の取得と切り出し ---
@st.cache_data(show_spinner=False)
def get_sliced_remote_wav(file_url, original_start, original_end):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception("ファイルのダウンロードに失敗したぜ。")
        
    wav_io = io.BytesIO(response.content)
    sample_rate, data = wavfile.read(wav_io)
    total_duration = len(data) / sample_rate
    
    start_sec = max(0.0, original_start - 1.5)
    end_sec = min(total_duration, original_end + 1.5)
    
    start_frame = int(start_sec * sample_rate)
    end_frame = int(end_sec * sample_rate)
    
    sliced_data = data[start_frame:end_frame]
    
    out_wav = io.BytesIO()
    wavfile.write(out_wav, sample_rate, sliced_data)
    out_wav.seek(0)
    
    return out_wav, start_sec, end_sec

# --- 2.5. 日付情報の自動抽出と曜日の計算 ---
def extract_date(filename):
    match = re.match(r'^(\d{6})', filename)
    if match:
        try:
            dt = datetime.strptime(match.group(1), '%y%m%d')
            weekdays = ['月', '火', '水', '木', '金', '土', '日']
            weekday_str = weekdays[dt.weekday()]
            return f"{dt.strftime('%Y/%m/%d')} ({weekday_str})"
        except ValueError:
            return "不明な日付"
    return "不明な日付"

# --- 2.8. 画面遷移のコントロール（Session State） ---
if 'page' not in st.session_state: st.session_state.page = 'main'
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'selected_bird' not in st.session_state: st.session_state.selected_bird = None
if 'selected_loc' not in st.session_state: st.session_state.selected_loc = None
if 'loaded_audio' not in st.session_state: st.session_state.loaded_audio = set()

def load_audio_clip(audio_key):
    st.session_state.loaded_audio.add(audio_key)

def go_to_date_detail(date_str):
    st.session_state.page = 'date_detail'
    st.session_state.selected_date = date_str

def go_to_bird_detail(bird_name):
    st.session_state.page = 'bird_detail'
    st.session_state.selected_bird = bird_name

def go_to_loc_detail(loc_name):
    st.session_state.page = 'loc_detail'
    st.session_state.selected_loc = loc_name

def go_to_main():
    st.session_state.page = 'main'
    st.session_state.selected_date = None
    st.session_state.selected_bird = None
    st.session_state.selected_loc = None

# --- 3. メインUI ---

# タイトルがスマホで折り返さないように、st.titleではなくHTMLでスマートなサイズに調整
st.markdown("<h2 style='font-size: 26px; font-weight: bold; padding-top: 10px;'>🎧 Ambient Bird Log 🐦</h2>", unsafe_allow_html=True)

# 🔥 GEʍlNEʍ's CSS Hack: グリッドレイアウト用の究極のスタイル
st.markdown("""
    <style>
    /* 要素間の隙間を極限まで詰める */
    div[data-testid="stVerticalBlock"] { gap: 0rem; }
    
    /* 📸 サムネイル画像を強制的に「正方形（1:1）」にし、はみ出た部分を美しくカット */
    img { 
        border-radius: 6px; 
        object-fit: cover !important; 
        aspect-ratio: 1 / 1 !important; 
        width: 100% !important; 
        margin-bottom: 5px;
    }
    
    /* 🔘 ボタン内のテキストを小さくして、狭いカラム内に収める */
    button p {
        font-size: 11px !important;
    }
    </style>
""", unsafe_allow_html=True)

try:
    response_all = supabase.table("detections").select("*").execute()
    response_master = supabase.table("bird_master").select("*").execute()
    bird_images = {row['common_name']: row['image_url'] for row in response_master.data} if response_master.data else {}
    
    if response_all.data:
        df_all = pd.DataFrame(response_all.data)
        df_all['record_date'] = df_all['wav_filename'].apply(extract_date)
        
        # --- A. メインページ（タブ表示） ---
        if st.session_state.page == 'main':
            bird_names = sorted(df_all['common_name'].dropna().unique().tolist())
            is_admin = st.query_params.get("admin") == "true"
            
            if is_admin:
                tab_bird, tab_location, tab_admin = st.tabs(["🐦 鳥から探す", "📍 場所から探す", "⚙️ 画像管理"])
            else:
                tab_bird, tab_location = st.tabs(["🐦 鳥から探す", "📍 場所から探す"])

            # 【タブ1：鳥から探す】 (🔥 3カラムグリッドUI 🔥)
            with tab_bird:
                search_query = st.text_input(
                    "検索窓", 
                    label_visibility="collapsed", 
                    placeholder="和名で検索" 
                )
                st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
                
                bird_counts = df_all['common_name'].value_counts()
                
                # 検索ワードに引っかかった鳥だけを抽出
                filtered_birds = {name: count for name, count in bird_counts.items() if not search_query or search_query in name}
                
                # 🛠️ 3つのカラム（列）を作成
                cols = st.columns(3)
                
                # 抽出した鳥を3つのカラムに順番に振り分けていくループ処理
                for i, (bird_name, count) in enumerate(filtered_birds.items()):
                    col_idx = i % 3  # 0, 1, 2, 0, 1, 2... と順番にカラムを指定する算術ハック
                    
                    with cols[col_idx]:
                        # コンテナを使って要素のまとまりを作る
                        with st.container():
                            img_url = bird_images.get(bird_name)
                            if img_url:
                                st.image(img_url, use_container_width=True)
                            else:
                                st.markdown(
                                    "<div style='background-color:#1E1E1E; border-radius:6px; aspect-ratio: 1/1; display:flex; align-items:center; justify-content:center; margin-bottom: 5px;'><span style='color:#8E8E93; font-size:10px;'>No Img</span></div>", 
                                    unsafe_allow_html=True
                                )
                            
                            # 鳥の名前ボタン（幅いっぱいに広げる）
                            if st.button(f"{bird_name}", key=f"btn_bird_{bird_name}", use_container_width=True):
                                go_to_bird_detail(bird_name)
                                st.rerun()
                        
                        # 行の間に少しだけ縦の隙間を空ける
                        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

            # 【タブ2：場所から探す】
            with tab_location:
                st.markdown("### 🗺️ 場所から探す")
                df_loc = df_all.dropna(subset=['latitude', 'longitude'])
                if not df_loc.empty:
                    df_map = df_loc[['latitude', 'longitude', 'location_name']].drop_duplicates(subset=['latitude', 'longitude'])
                    st.map(df_map, latitude='latitude', longitude='longitude', color='#39FF14', size=150)
                    st.divider()
                    st.markdown("**📍 過去の記録（場所別）**")
                    unique_locations = sorted(df_loc['location_name'].unique().tolist())
                    selected_loc = st.selectbox("場所を選択してくれ:", unique_locations, label_visibility="collapsed")
                    if selected_loc:
                        loc_filtered = df_loc[df_loc['location_name'] == selected_loc].sort_values(by='record_date', ascending=False)
                        dates_at_loc = loc_filtered['record_date'].unique()
                        for date_str in dates_at_loc:
                            count = len(loc_filtered[loc_filtered['record_date'] == date_str])
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"### 📅 {date_str}")
                                    st.caption(f"🎧 検出: {count} 件")
                                with col2:
                                    st.button("詳細", on_click=go_to_date_detail, args=(date_str,), key=f"btn_{selected_loc}_{date_str}")
                            st.divider()

            # 【タブ3：画像管理（管理者ステルス用）】
            if is_admin:
                with tab_admin:
                    st.markdown("### 📷 野鳥画像のアップロード")
                    st.info("自らのレンズで捉えた最高のShotを、図鑑に登録するぜ。")
                    upload_bird = st.selectbox("画像を登録する鳥を選んでくれ:", bird_names, key="upload_select")
                    uploaded_file = st.file_uploader("画像をドロップするか選択してくれ (JPG/PNG)", type=["jpg", "jpeg", "png"])
                    
                    if uploaded_file is not None and upload_bird:
                        if st.button("🚀 Cloudへアップロード", use_container_width=True):
                            with st.spinner("Uploading to Supabase..."):
                                try:
                                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                                    file_ext = uploaded_file.name.split('.')[-1]
                                    safe_file_name = f"img_{timestamp}_{uuid.uuid4().hex[:8]}.{file_ext}"
                                    
                                    supabase.storage.from_("bird-images").upload(safe_file_name, uploaded_file.getvalue())
                                    public_image_url = supabase.storage.from_("bird-images").get_public_url(safe_file_name)
                                    
                                    supabase.table("bird_master").upsert({
                                        "common_name": upload_bird,
                                        "image_url": public_image_url
                                    }).execute()
                                    
                                    st.success(f"🔥 {upload_bird} の画像をCloudに刻み込んだぜ！")
                                except Exception as e:
                                    st.error(f"アップロードに失敗したぜ: {e}")

        # --- B. 鳥の詳細ページ (Page 1-2) ---
        elif st.session_state.page == 'bird_detail':
            st.button("⬅️ メインに戻る", on_click=go_to_main)
            
            target_bird = st.session_state.selected_bird
            bird_data = df_all[df_all['common_name'] == target_bird].sort_values(by='confidence', ascending=False)
            
            if not bird_data.empty:
                scientific_name = bird_data.iloc[0]['scientific_name']
                st.markdown(f"## 🐦 {target_bird}")
                st.caption(f"学名: *{scientific_name}*")
                
                img_response = supabase.table("bird_master").select("image_url").eq("common_name", target_bird).execute()
                if img_response.data and img_response.data[0]['image_url']:
                    st.image(img_response.data[0]['image_url'], use_container_width=True)
                else:
                    st.info("🖼️ まだ画像が登録されていないぜ。「⚙️ 画像管理」タブからUploadしてくれ。")
                    
                st.divider()
                
                loop_idx = 0
                for index, row in bird_data.iterrows():
                    with st.container():
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        duration = round(float(row['end_sec']) - float(row['start_sec']), 1)
                        confidence_pct = int(row['confidence'] * 100)
                        
                        col_meta1, col_meta2 = st.columns([1, 1])
                        with col_meta1:
                            st.markdown(f"**信頼度:** `{confidence_pct}%`")
                            st.markdown(f"**再生時間:** `{duration}秒`")
                        with col_meta2:
                            st.button(f"📅 {row['record_date']}", on_click=go_to_date_detail, args=(row['record_date'],), key=f"link_date_{index}")
                            loc_name = row['location_name'] if pd.notna(row['location_name']) else "場所不明"
                            # 場所への直接ワープリンク
                            st.button(f"📍 {loc_name}", on_click=go_to_loc_detail, args=(loc_name,), key=f"link_loc_{index}")
                        
                        # 遅延読み込みロジック
                        audio_key = f"audio_{wav_filename}_{row['start_sec']}"
                        if loop_idx == 0 or audio_key in st.session_state.loaded_audio:
                            try:
                                with st.spinner("Loading Audio..."):
                                    sliced_audio, actual_start, actual_end = get_sliced_remote_wav(
                                        public_url, float(row['start_sec']), float(row['end_sec'])
                                    )
                                st.audio(sliced_audio, format="audio/wav")
                            except Exception as e:
                                st.error(f"ロードエラー: {e}")
                        else:
                            st.button("🔊 再生データをロード", on_click=load_audio_clip, args=(audio_key,), key=f"btn_load_{audio_key}")
                            
                    st.divider()
                    loop_idx += 1

        # --- C. 記録日詳細ページ (Page 3) ---
        elif st.session_state.page == 'date_detail':
            st.button("⬅️ メインに戻る", on_click=go_to_main)
            target_date = st.session_state.selected_date
            st.markdown(f"## 📅 {target_date} の記録")
            day_data = df_all[df_all['record_date'] == target_date]
            
            if not day_data.empty:
                visited_locations = day_data['location_name'].dropna().unique()
                loc_text = "、".join(visited_locations) if len(visited_locations) > 0 else "場所不明"
                st.info(f"📍 **その日に行った場所:** {loc_text}")
                st.markdown(f"### 🐦 その日の鳥 (計 {len(day_data)} 件)")
                day_data_sorted = day_data.sort_values(by='confidence', ascending=False)
                
                loop_idx = 0
                for index, row in day_data_sorted.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        
                        with col1:
                            confidence_pct = int(row['confidence'] * 100)
                            st.button(f"**{row['common_name']}**", on_click=go_to_bird_detail, args=(row['common_name'],), key=f"link_bird_date_{index}")
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                            
                        with col2:
                            audio_key = f"audio_{wav_filename}_{row['start_sec']}"
                            if loop_idx == 0 or audio_key in st.session_state.loaded_audio:
                                try:
                                    with st.spinner("Loading..."):
                                        sliced_audio, actual_start, actual_end = get_sliced_remote_wav(
                                            public_url, float(row['start_sec']), float(row['end_sec'])
                                        )
                                    st.audio(sliced_audio, format="audio/wav")
                                except Exception as e:
                                    st.error(f"ロードエラー: {e}")
                            else:
                                st.button("🔊 再生データをロード", on_click=load_audio_clip, args=(audio_key,), key=f"btn_load_date_{audio_key}")
                                
                    st.divider()
                    loop_idx += 1

        # --- D. 場所詳細ページ (Page 1-2) ---
        elif st.session_state.page == 'loc_detail':
            st.button("⬅️ メインに戻る", on_click=go_to_main)
            
            target_loc = st.session_state.selected_loc
            st.markdown(f"## 📍 {target_loc} の記録")
            
            loc_data = df_all[df_all['location_name'] == target_loc].sort_values(by='record_date', ascending=False)
            
            if not loc_data.empty:
                dates_at_loc = loc_data['record_date'].unique()
                for date_str in dates_at_loc:
                    count = len(loc_data[loc_data['record_date'] == date_str])
                    with st.container():
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.markdown(f"### 📅 {date_str}")
                            st.caption(f"🎧 検出: {count} 件")
                        with col2:
                            st.button("詳細", on_click=go_to_date_detail, args=(date_str,), key=f"btn_loc_detail_{target_loc}_{date_str}")
                    st.divider()
            else:
                st.warning("この場所のデータは見つからなかったぜ。")

    else:
        st.warning("クラウドのデータベースにデータがないぜ。")
        
except Exception as e:
    st.error(f"システムエラー: {e}")