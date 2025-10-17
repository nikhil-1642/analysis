with tab3:
    st.header("📝 Text Analysis & Visualization")

    # Hardcoded sample stories
    stories = [
        # --- Your existing story list ---
        """In a remote kingdom nestled between jagged mountains...""",
        """During the bustling era of the 1920s, in a city that never slept...""",
        """On a distant exoplanet, where the sky shimmered...""",
        """In the neon-lit heart of Tokyo, young coder Akira...""",
        """Deep in the Amazon rainforest, a team of scientists..."""
    ]

    # Initialize text area session state
    if "text_area" not in st.session_state:
        st.session_state.text_area = ""

    # Random story button
    if st.button("🎲 Random Story"):
        st.session_state.text_area = random.choice(stories)

    # Text input area
    st.session_state.text_area = st.text_area(
        "📜 Paste or modify your text here:",
        value=st.session_state.text_area,
        height=250
    )

    # Analyze button
    if st.button("Analyze Text 🚀"):
        text = st.session_state.text_area.strip()

        if text:
            blob = TextBlob(text)

            # -----------------------------
            # 1️⃣ Language Detection
            # -----------------------------
            lang = blob.detect_language()
            if lang != 'en':
                st.info(f"🌐 Detected language: {lang}. Translating to English...")
                blob = blob.translate(to='en')

            # -----------------------------
            # 2️⃣ Sentiment Analysis
            # -----------------------------
            polarity = round(blob.sentiment.polarity, 3)
            subjectivity = round(blob.sentiment.subjectivity, 3)

            st.subheader("💬 Sentiment Analysis")
            sentiment = (
                "😊 Positive" if polarity > 0.1 else
                "😐 Neutral" if -0.1 <= polarity <= 0.1 else
                "😠 Negative"
            )
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Polarity:** {polarity}")
            st.write(f"**Subjectivity:** {subjectivity}")

            # Polarity bar
            st.progress((polarity + 1) / 2)

            # -----------------------------
            # 3️⃣ POS Tagging
            # -----------------------------
            words_and_tags = blob.tags
            nouns = [w for w, t in words_and_tags if t.startswith('NN')]
            verbs = [w for w, t in words_and_tags if t.startswith('VB')]
            adjectives = [w for w, t in words_and_tags if t.startswith('JJ')]
            adverbs = [w for w, t in words_and_tags if t.startswith('RB')]

            # -----------------------------
            # 4️⃣ WordCloud Function
            # -----------------------------
            def make_wordcloud(words, color):
                if not words:
                    return None
                wc = WordCloud(width=500, height=400, background_color='black', colormap=color).generate(" ".join(words))
                fig, ax = plt.subplots()
                ax.imshow(wc, interpolation='bilinear')
                ax.axis("off")
                return fig

            # -----------------------------
            # 5️⃣ Display WordClouds
            # -----------------------------
            col1, col2 = st.columns(2)
            col3, col4 = st.columns(2)

            with col1:
                st.markdown("### 🧠 Nouns")
                fig = make_wordcloud(nouns, "plasma")
                if fig: st.pyplot(fig)
            with col2:
                st.markdown("### ⚡ Verbs")
                fig = make_wordcloud(verbs, "inferno")
                if fig: st.pyplot(fig)
            with col3:
                st.markdown("### 🎨 Adjectives")
                fig = make_wordcloud(adjectives, "cool")
                if fig: st.pyplot(fig)
            with col4:
                st.markdown("### 💨 Adverbs")
                fig = make_wordcloud(adverbs, "magma")
                if fig: st.pyplot(fig)

            # -----------------------------
            # 6️⃣ POS Counts
            # -----------------------------
            st.markdown("### 📊 POS Counts")
            st.write({
                "Nouns": len(nouns),
                "Verbs": len(verbs),
                "Adjectives": len(adjectives),
                "Adverbs": len(adverbs)
            })

            # -----------------------------
            # 7️⃣ Download Analysis
            # -----------------------------
            report = f"""
            TEXT ANALYSIS REPORT
            -------------------------
            Sentiment: {sentiment}
            Polarity: {polarity}
            Subjectivity: {subjectivity}

            Nouns: {len(nouns)} | Verbs: {len(verbs)} | Adjectives: {len(adjectives)} | Adverbs: {len(adverbs)}
            -------------------------
            Most frequent Nouns: {', '.join(nouns[:10])}
            Most frequent Verbs: {', '.join(verbs[:10])}
            """
            st.download_button("⬇️ Download Analysis Report", report, file_name="text_analysis.txt")

        else:
            st.warning("Please paste or select some text first.")
