mkdir -p ~/.streamlit/

echo "[theme]
base='light'
primaryColor='#86BC24'
secondaryBackgroundColor='#0F0808'
textColor='#9c9da0'
[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
