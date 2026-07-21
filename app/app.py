import streamlit as st
import pandas as pd
import requests
import io
import os
from dotenv import load_dotenv
from supabase import create_client, Client
from pydub import AudioSegment

# --- 1. 初期設定と環境変数の読み込み ---
st.set_page_config(page_title="Ambient Bird Log", page_icon="🐦", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(dotenv_path=os.path.join(BASE_DIR, ".env"))

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")
BUCKET_NAME = "bird-wav"

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Supabaseの環境変数が見つからないぜ。.envファイルを確認してくれ。")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 🔥 GEʍlNEʍ's CSS Hack (トップ余白の最適化＆レスポンシブ版) ---
st.markdown("""
    <style>
    /* ネイティブヘッダーを避けつつ余白を削る黄金比 */
    .block-container {
        padding-top: 3.5rem !important; 
        padding-bottom: 1rem !important;
    }
    /* 既存のレイアウト調整 */
    div[data-testid="stVerticalBlock"] { gap: 0rem; }
    img { 
        border-radius: 6px; 
        object-fit: cover !important; 
        aspect-ratio: 1 / 1 !important; 
        width: 100% !important; 
        margin-bottom: 4px !important;
    }
    @media (max-width: 640px) {
        div[data-testid="stHorizontalBlock"] { flex-direction: row !important; flex-wrap: nowrap !important; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] { min-width: 0 !important; padding: 0 3px !important; }
        button p { font-size: 10px !important; }
    }
    button p { font-size: 12px !important; }
    </style>
""", unsafe_allow_html=True)

# --- 2. 状態管理（Session State） ---
if 'page' not in st.session_state: st.session_state.page = 'main'
if 'selected_date' not in st.session_state: st.session_state.selected_date = None
if 'selected_bird' not in st.session_state: st.session_state.selected_bird = None
if 'selected_loc' not in st.session_state: st.session_state.selected_loc = None
if 'loaded_audio' not in st.session_state: st.session_state.loaded_audio = set()

def go_to_main():
    st.session_state.page = 'main'
    st.session_state.selected_date = None
    st.session_state.selected_bird = None
    st.session_state.selected_loc = None

def go_to_date_detail(date_str):
    st.session_state.page = 'date_detail'
    st.session_state.selected_date = date_str

def go_to_bird_detail(bird_name):
    st.session_state.page = 'bird_detail'
    st.session_state.selected_bird = bird_name

def go_to_loc_detail(loc_name):
    st.session_state.page = 'loc_detail'
    st.session_state.selected_loc = loc_name

# --- 3. データ取得と処理関数 ---
@st.cache_data(ttl=600, show_spinner=False)
def load_all_data():
    try:
        response = supabase.table("detections").select("*").execute()
        df = pd.DataFrame(response.data)
        if not df.empty and 'start_sec' in df.columns:
            # 日付カラムの作成
            df['record_date'] = df['wav_filename'].apply(lambda x: "20" + x[:2] + "/" + x[2:4] + "/" + x[4:6] if isinstance(x, str) and len(x) > 6 else "Unknown")
        return df
    except Exception as e:
        st.error(f"データ読み込みエラー: {e}")
        return pd.DataFrame()

# 🔥 pydubを使ったMP3での切り出し
@st.cache_data(show_spinner=False)
def get_sliced_remote_audio(file_url, original_start, original_end):
    response = requests.get(file_url)
    if response.status_code != 200:
        raise Exception("ファイルのダウンロードに失敗したぜ。")
        
    audio = AudioSegment.from_file(io.BytesIO(response.content))
    start_ms = max(0, int((original_start - 1.5) * 1000))
    end_ms = min(len(audio), int((original_end + 1.5) * 1000))
    
    sliced_audio = audio[start_ms:end_ms]
    out_io = io.BytesIO()
    sliced_audio.export(out_io, format="mp3", bitrate="192k")
    out_io.seek(0)
    
    return out_io, start_ms / 1000.0, end_ms / 1000.0

# --- 4. メインアプリケーション ---
def main():
    df_all = load_all_data()

    # ヘッダー
    st.markdown("<h2 style='text-align: center; margin-bottom: 0;'>🎧 Ambient Bird Log 🐦</h2>", unsafe_allow_html=True)

    if df_all.empty:
        st.warning("データがまだ登録されていないようだぜ。")
        is_admin = st.query_params.get("admin") == "true"
        if is_admin:
            tab_data, = st.tabs(["📁 データ登録"])
            with tab_data:
                st.markdown("### 📁 データの登録")
        return

    # ==========================================
    # メイン画面
    # ==========================================
    if st.session_state.page == 'main':
        is_admin = st.query_params.get("admin") == "true"
        
        if is_admin:
            tab_bird, tab_location, tab_admin, tab_data = st.tabs(["🐦 鳥から探す", "📍 場所から探す", "⚙️ 画像管理", "📁 データ登録"])
        else:
            tab_bird, tab_location = st.tabs(["🐦 鳥から探す", "📍 場所から探す"])

        # --- タブ: 鳥から探す ---
        # --- タブ: 鳥から探す ---
        with tab_bird:
            st.markdown("### 🐦 鳥から探す")
            unique_birds = sorted(df_all['common_name'].dropna().unique().tolist())
            selected_main_bird = st.selectbox("和名で検索", ["選択してください"] + unique_birds, label_visibility="collapsed")
            
            if selected_main_bird != "選択してください":
                go_to_bird_detail(selected_main_bird)
                st.rerun()
            
            # ==========================================
            # 🔥 ここから下を新規追加：サムネイルギャラリー
            # ==========================================
            st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
            
            # 画像が保存されているバケット名（※実際の環境に合わせて変更してくれ）
            IMAGE_BUCKET = "bird-images" 
            
            try:
                # ストレージから画像一覧を抽出
                img_list = supabase.storage.from_(IMAGE_BUCKET).list()
                img_names = [img['name'] for img in img_list if img['name'] != '.emptyFolderPlaceholder']
                
                # 鳥の和名と画像ファイルをマッチング
                bird_img_map = {}
                for bird in unique_birds:
                    for img_name in img_names:
                        # 画像ファイル名が鳥の名前を含んでいれば採用
                        if bird in img_name: 
                            bird_img_map[bird] = img_name
                            break
                            
                if bird_img_map:
                    # 3カラムのグリッドレイアウトを作成
                    cols = st.columns(3)
                    col_idx = 0
                    for bird, img_file in bird_img_map.items():
                        with cols[col_idx % 3]:
                            # 画像のパブリックURLを取得して表示
                            img_url = supabase.storage.from_(IMAGE_BUCKET).get_public_url(img_file)
                            st.image(img_url)
                            # 画像の下に詳細画面へのジャンプボタンを配置
                            if st.button(bird, key=f"btn_thumb_{bird}", use_container_width=True):
                                go_to_bird_detail(bird)
                                st.rerun()
                        col_idx += 1
            except Exception as e:
                # バケットが存在しない等のエラー時はサイレントにスキップ
                pass

            st.divider()
            st.markdown("**📅 過去の記録（日付別）**")
            # ... (以下既存のコードが続く) ...
                
            st.divider()
            st.markdown("**📅 過去の記録（日付別）**")
            unique_dates = sorted(df_all['record_date'].dropna().unique().tolist(), reverse=True)
            for d in unique_dates:
                if st.button(f"📅 {d}", use_container_width=True, key=f"btn_date_{d}"):
                    go_to_date_detail(d)
                    st.rerun()

        # --- タブ: 場所から探す ---
        with tab_location:
            st.markdown("### 🗺️ 場所から探す")
            df_loc = df_all.dropna(subset=['latitude', 'longitude'])
            if not df_loc.empty:
                df_map = df_loc[['latitude', 'longitude', 'location_name']].drop_duplicates(subset=['latitude', 'longitude']).copy()
                
                # 🔥 GEʍlNEʍ Hack: マップの縮尺を適切に保つためのダミーポイント
                lat_min, lat_max = df_map['latitude'].min(), df_map['latitude'].max()
                lon_min, lon_max = df_map['longitude'].min(), df_map['longitude'].max()
                
                lat_pad = max((lat_max - lat_min) * 0.8, 0.05)
                lon_pad = max((lon_max - lon_min) * 0.8, 0.05)
                
                df_map['dot_color'] = '#39FF14'
                df_map['dot_size'] = 150
                
                dummy_data = pd.DataFrame({
                    'latitude': [lat_min - lat_pad, lat_max + lat_pad],
                    'longitude': [lon_min - lon_pad, lon_max + lon_pad],
                    'location_name': ['dummy', 'dummy'],
                    'dot_color': ['#39FF1405', '#39FF1405'], 
                    'dot_size': [1, 1] 
                })
                
                df_map_padded = pd.concat([df_map, dummy_data], ignore_index=True)
                st.map(df_map_padded, latitude='latitude', longitude='longitude', color='dot_color', size='dot_size', height=250)
                
                st.divider()
                st.markdown("**📍 過去の記録（場所別）**")
                unique_locations = sorted(df_loc['location_name'].unique().tolist())
                for loc in unique_locations:
                    if st.button(f"📍 {loc}", use_container_width=True, key=f"btn_loc_{loc}"):
                        go_to_loc_detail(loc)
                        st.rerun()

        # --- タブ: データ登録 (管理者のみ) ---
        if is_admin:
            with tab_data:
                st.markdown("### 📁 データの登録")
                uploaded_csvs = st.file_uploader("CSVファイルを選択", type=['csv'], accept_multiple_files=True)
                uploaded_wavs = st.file_uploader("WAVファイルを選択 (自動でMP3に圧縮されます)", type=['wav'], accept_multiple_files=True)
                loc_name_input = st.text_input("場所の名前")
                lat_input = st.number_input("緯度 (Latitude)", format="%.6f")
                lon_input = st.number_input("経度 (Longitude)", format="%.6f")
                
                if st.button("クラウドへ登録", type="primary", use_container_width=True):
                    if not uploaded_csvs:
                        st.error("CSVファイルが必要だぜ。")
                    else:
                        with st.spinner("データを処理中だぜ..."):
                            try:
                                # WAVをMP3に圧縮してアップロード
                                if uploaded_wavs:
                                    for wav_file in uploaded_wavs:
                                        audio = AudioSegment.from_file(wav_file)
                                        mp3_io = io.BytesIO()
                                        audio.export(mp3_io, format="mp3", bitrate="192k")
                                        mp3_io.seek(0)
                                        
                                        mp3_filename = wav_file.name.rsplit('.', 1)[0] + ".mp3"
                                        supabase.storage.from_(BUCKET_NAME).upload(
                                            mp3_filename, 
                                            mp3_io.read(), 
                                            file_options={"content-type": "audio/mpeg", "upsert": "true"}
                                        )
                                    st.success(f"🎵 {len(uploaded_wavs)} 個の音声をMP3に圧縮してアップロードしたぜ！")

                                if uploaded_csvs:
                                    all_data = []
                                    for uploaded_csv in uploaded_csvs:
                                        df_csv = pd.read_csv(uploaded_csv)
                                        df_csv = df_csv.rename(columns={'Start (s)': 'start_sec', 'End (s)': 'end_sec', 'Scientific name': 'scientific_name', 'Common name': 'common_name', 'Confidence': 'confidence'})
                                        if 'File' in df_csv.columns:
                                            df_csv['wav_filename'] = df_csv['File'].apply(lambda x: os.path.basename(str(x)).rsplit('.', 1)[0] + ".mp3")
                                            df_csv = df_csv.drop(columns=['File'])
                                        df_csv['location_name'] = loc_name_input
                                        df_csv['latitude'] = lat_input
                                        df_csv['longitude'] = lon_input
                                        all_data.append(df_csv)
                                        
                                    final_df = pd.concat(all_data, ignore_index=True)
                                    supabase.table("detections").insert(final_df.to_dict('records')).execute()
                                    st.success("✅ CSVデータのデータベース登録が完了したぜ！")
                                    st.cache_data.clear()
                            except Exception as e:
                                st.error(f"エラーが発生したぜ: {e}")

    # ==========================================
    # 詳細画面: 日付 (date_detail)
    # ==========================================
    elif st.session_state.page == 'date_detail':
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            st.button("🐦 鳥から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_date_1")
        with col_nav2:
            st.button("📍 場所から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_date_2")
        st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #444;'>", unsafe_allow_html=True)
        
        target_date = st.session_state.selected_date
        count_key = f"display_count_date_{target_date}"
        if count_key not in st.session_state: st.session_state[count_key] = 10
            
        st.markdown(f"## 📅 {target_date} の記録")
        day_data = df_all[df_all['record_date'] == target_date]
        
        if not day_data.empty:
            visited_locations = day_data['location_name'].dropna().unique()
            loc_text = "、".join(visited_locations) if len(visited_locations) > 0 else "場所不明"
            st.info(f"📍 **その日に行った場所:** {loc_text}")
            
            # フィルター群
            min_confidence = st.slider("信頼度で絞り込む", min_value=0, max_value=100, value=60, format="%d%%", key=f"slider_conf_date_{target_date}")
            day_data = day_data[day_data['confidence'] >= (min_confidence / 100.0)]
            
            if not day_data.empty:
                available_birds = ["鳥で絞り込む (すべて)"] + sorted(day_data['common_name'].dropna().unique().tolist())
                selected_bird_filter = st.selectbox("鳥を選択", available_birds, key=f"filter_bird_date_{target_date}", label_visibility="collapsed")
                if selected_bird_filter != "鳥で絞り込む (すべて)":
                    day_data = day_data[day_data['common_name'] == selected_bird_filter]
            
            st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)
            
            if day_data.empty:
                st.warning("指定した条件に一致する録音データは見つからなかったぜ。")
            else:
                st.markdown(f"### 🐦 その日の鳥 (計 {len(day_data)} 件)")
                day_data_sorted = day_data.sort_values(by='confidence', ascending=False)
                
                current_limit = st.session_state[count_key]
                day_data_to_show = day_data_sorted.head(current_limit)
                
                for index, row in day_data_to_show.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        
                        with col1:
                            confidence_pct = int(row['confidence'] * 100)
                            st.button(f"**{row['common_name']}**", on_click=go_to_bird_detail, args=(row['common_name'],), key=f"link_bird_date_{index}")
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                            
                        with col2:
                            try:
                                with st.spinner("Loading..."):
                                    sliced_audio, actual_start, actual_end = get_sliced_remote_audio(public_url, float(row['start_sec']), float(row['end_sec']))
                                st.audio(sliced_audio, format="audio/mpeg")
                            except Exception as e:
                                st.error(f"ロードエラー")
                    st.divider()
                
                if current_limit < len(day_data):
                    if st.button("🔽 さらに10件読み込む", use_container_width=True, key=f"btn_load_more_date_{target_date}"):
                        st.session_state[count_key] += 10
                        st.rerun()

    # ==========================================
    # 詳細画面: 鳥 (bird_detail)
    # ==========================================
    elif st.session_state.page == 'bird_detail':
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            st.button("🐦 鳥から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_bird_1")
        with col_nav2:
            st.button("📍 場所から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_bird_2")
        st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #444;'>", unsafe_allow_html=True)
        
        target_bird = st.session_state.selected_bird
        count_key = f"display_count_bird_{target_bird}"
        if count_key not in st.session_state: st.session_state[count_key] = 10
            
        st.markdown(f"## 🐦 {target_bird} の記録")
        bird_data = df_all[df_all['common_name'] == target_bird]
        
        if not bird_data.empty:
            seen_locations = bird_data['location_name'].dropna().unique()
            loc_text = "、".join(seen_locations) if len(seen_locations) > 0 else "場所不明"
            st.info(f"📍 **見つけた場所:** {loc_text}")
            
            # フィルター群
            min_confidence = st.slider("信頼度で絞り込む", min_value=0, max_value=100, value=60, format="%d%%", key=f"slider_conf_bird_{target_bird}")
            bird_data = bird_data[bird_data['confidence'] >= (min_confidence / 100.0)]
            
            if not bird_data.empty:
                available_dates = ["日付で絞り込む (すべて)"] + sorted(bird_data['record_date'].dropna().unique().tolist(), reverse=True)
                selected_date_filter = st.selectbox("日付を選択", available_dates, key=f"filter_date_bird_{target_bird}", label_visibility="collapsed")
                if selected_date_filter != "日付で絞り込む (すべて)":
                    bird_data = bird_data[bird_data['record_date'] == selected_date_filter]
            
            st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)
            
            if bird_data.empty:
                st.warning("指定した条件に一致する録音データは見つからなかったぜ。")
            else:
                st.markdown(f"### 📅 検出リスト (計 {len(bird_data)} 件)")
                bird_data_sorted = bird_data.sort_values(by='record_date', ascending=False)
                
                current_limit = st.session_state[count_key]
                bird_data_to_show = bird_data_sorted.head(current_limit)
                
                for index, row in bird_data_to_show.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        
                        with col1:
                            confidence_pct = int(row['confidence'] * 100)
                            st.button(f"**📅 {row['record_date']}**", on_click=go_to_date_detail, args=(row['record_date'],), key=f"link_date_bird_{index}")
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                            
                        with col2:
                            try:
                                with st.spinner("Loading..."):
                                    sliced_audio, actual_start, actual_end = get_sliced_remote_audio(public_url, float(row['start_sec']), float(row['end_sec']))
                                st.audio(sliced_audio, format="audio/mpeg")
                            except Exception as e:
                                st.error(f"ロードエラー")
                    st.divider()
                
                if current_limit < len(bird_data):
                    if st.button("🔽 さらに10件読み込む", use_container_width=True, key=f"btn_load_more_bird_{target_bird}"):
                        st.session_state[count_key] += 10
                        st.rerun()

    # ==========================================
    # 詳細画面: 場所 (loc_detail)
    # ==========================================
    elif st.session_state.page == 'loc_detail':
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            st.button("🐦 鳥から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_loc_1")
        with col_nav2:
            st.button("📍 場所から探す", on_click=go_to_main, type="tertiary", use_container_width=True, key="nav_loc_2")
        st.markdown("<hr style='margin-top: -10px; margin-bottom: 16px; border-color: #444;'>", unsafe_allow_html=True)
        
        target_loc = st.session_state.selected_loc
        count_key = f"display_count_loc_{target_loc}"
        if count_key not in st.session_state: st.session_state[count_key] = 10
            
        st.markdown(f"## 📍 {target_loc} の記録")
        loc_data = df_all[df_all['location_name'] == target_loc]
        
        if not loc_data.empty:
            seen_birds = loc_data['common_name'].dropna().unique()
            st.info(f"🐦 **ここで見つけた鳥:** {len(seen_birds)} 種類")
            
            # フィルター群
            min_confidence = st.slider("信頼度で絞り込む", min_value=0, max_value=100, value=60, format="%d%%", key=f"slider_conf_loc_{target_loc}")
            loc_data = loc_data[loc_data['confidence'] >= (min_confidence / 100.0)]
            
            if not loc_data.empty:
                available_birds = ["鳥で絞り込む (すべて)"] + sorted(loc_data['common_name'].dropna().unique().tolist())
                selected_bird_filter = st.selectbox("鳥を選択", available_birds, key=f"filter_bird_loc_{target_loc}", label_visibility="collapsed")
                if selected_bird_filter != "鳥で絞り込む (すべて)":
                    loc_data = loc_data[loc_data['common_name'] == selected_bird_filter]
            
            st.markdown("<div style='min-height: 10px;'></div>", unsafe_allow_html=True)
            
            if loc_data.empty:
                st.warning("指定した条件に一致する録音データは見つからなかったぜ。")
            else:
                st.markdown(f"### 🐦 検出リスト (計 {len(loc_data)} 件)")
                loc_data_sorted = loc_data.sort_values(by='confidence', ascending=False)
                
                current_limit = st.session_state[count_key]
                loc_data_to_show = loc_data_sorted.head(current_limit)
                
                for index, row in loc_data_to_show.iterrows():
                    with st.container():
                        col1, col2 = st.columns([3, 2])
                        wav_filename = row['wav_filename']
                        public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(wav_filename)
                        
                        with col1:
                            confidence_pct = int(row['confidence'] * 100)
                            st.button(f"**{row['common_name']}**", on_click=go_to_bird_detail, args=(row['common_name'],), key=f"link_bird_loc_{index}")
                            st.progress(row['confidence'], text=f"信頼度: {confidence_pct}%")
                            
                        with col2:
                            try:
                                with st.spinner("Loading..."):
                                    sliced_audio, actual_start, actual_end = get_sliced_remote_audio(public_url, float(row['start_sec']), float(row['end_sec']))
                                st.audio(sliced_audio, format="audio/mpeg")
                            except Exception as e:
                                st.error(f"ロードエラー")
                    st.divider()
                
                if current_limit < len(loc_data):
                    if st.button("🔽 さらに10件読み込む", use_container_width=True, key=f"btn_load_more_loc_{target_loc}"):
                        st.session_state[count_key] += 10
                        st.rerun()

if __name__ == "__main__":
    main()