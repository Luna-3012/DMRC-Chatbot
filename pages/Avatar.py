import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="Choose Your Avatar", layout="wide")

# Custom CSS for avatar grid layout
st.markdown("""
<style>
.avatar-container {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: flex-start;
    width: 100%;
    padding: 0.5em;
}

.avatar-image {
    background-color: black;     
    padding: 8px;                
    border-radius: 50%;          
    width: 110px;                
    height: 110px;
    object-fit: contain;
    margin-bottom: 0.5em;
}

.stButton > button {
    width: 80%;
    margin: 0 auto;
    display: block;
    font-size: 0.85em;
}
</style>
""", unsafe_allow_html=True)

st.title("👤 Choose Your Avatar!")

# Ensure assets directory exists
assets_dir = "assets"
if not os.path.exists(assets_dir):
    os.makedirs(assets_dir)

# Avatar configuration
AVATAR_COUNT = 30
AVATAR_SIZE = 150  # px

# Generate list of avatar file paths
avatars = [f"{assets_dir}/avatar{i}.png" for i in range(1, AVATAR_COUNT + 1)]

# Display avatars in grid layout
cols_per_row = 5
rows = [avatars[i:i+cols_per_row] for i in range(0, len(avatars), cols_per_row)]

for row in rows:
    cols = st.columns(cols_per_row)
    for col, avatar_path in zip(cols, row):
        with col:
            if os.path.exists(avatar_path):
                st.markdown('<div class="avatar-container">', unsafe_allow_html=True)
                st.image(avatar_path, output_format="PNG", use_container_width=False)
                if st.button("Select", key=avatar_path):
                    st.session_state.user_avatar = avatar_path
                    st.success("✔ Avatar selected!")
                    # Navigate back to main chat interface
                    home_page = Path("Home.py")
                    if home_page.exists():
                        st.switch_page(str(home_page))
                    else:
                        st.error("Home page not found.")
                st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🚇 DMRC Chatbot | Avatar Selection</p>
</div>
""", unsafe_allow_html=True)
