# app.py
import streamlit as st
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Initialize CLIP model
@st.cache_resource
def load_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

# Brand color palette for Aura Cosmetics
BRAND_COLORS = {
    "Lavender": [230, 230, 250],
    "Sage Green": [188, 212, 176],
    "Cream": [255, 253, 245],
    "Soft Pink": [255, 218, 224],
    "Sky Blue": [176, 224, 230]
}

def analyze_brand_consistency(image, model, processor):
    """Analyze if image matches Aura Cosmetics brand identity"""
    
    brand_descriptions = [
        "luxury pet care product with calming aesthetic",
        "natural and organic pet wellness",
        "soft pastel colors with premium feel",
        "happy pets in serene environment",
        "professional product photography",
        "wellness and tranquility theme",
        "high-quality pet care visuals"
    ]
    
    inputs = processor(
        text=brand_descriptions, 
        images=image, 
        return_tensors="pt", 
        padding=True
    )
    
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    
    # Get top 3 matches
    top_3_idx = probs[0].argsort(descending=True)[:3]
    top_matches = [(brand_descriptions[idx], float(probs[0][idx])) for idx in top_3_idx]
    
    return {
        "overall_score": float(probs.mean()),
        "top_matches": top_matches,
        "all_scores": {desc: float(score) for desc, score in zip(brand_descriptions, probs[0])}
    }

def analyze_audience_targeting(image, model, processor, age_group):
    """Check if image appeals to target demographic"""
    
    audience_descriptions = {
        "18-24 (Youth)": [
            "trendy pet content for social media",
            "fun and playful pet photography",
            "authentic lifestyle pet moment",
            "vibrant and energetic pet content",
            "shareable viral pet content"
        ],
        "25-44 (Core)": [
            "family pet wellness content",
            "professional pet care imagery",
            "trustworthy product demonstration",
            "clean and organized pet space",
            "lifestyle pet photography"
        ],
        "45+ (Premium)": [
            "premium luxury pet products",
            "sophisticated pet care visuals",
            "expert-endorsed pet wellness",
            "traditional quality imagery",
            "refined aesthetic"
        ]
    }
    
    descriptions = audience_descriptions.get(age_group, audience_descriptions["25-44 (Core)"])
    
    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    
    top_match_idx = probs[0].argmax()
    
    return {
        "audience_fit_score": float(probs.mean()),
        "best_match": descriptions[top_match_idx],
        "match_confidence": float(probs[0][top_match_idx])
    }

def analyze_platform_suitability(image, model, processor):
    """Check which platform this image suits best"""
    
    platform_descriptions = {
        "Instagram": "aesthetic lifestyle pet photography with perfect composition",
        "TikTok": "fun authentic pet content with movement and energy",
        "Facebook": "informative pet care content with clear messaging",
        "Pinterest": "inspirational pet wellness imagery worth saving",
        "LinkedIn": "professional pet industry content"
    }
    
    descriptions = list(platform_descriptions.values())
    platforms = list(platform_descriptions.keys())
    
    inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)
    
    platform_scores = {platform: float(score) for platform, score in zip(platforms, probs[0])}
    best_platform = max(platform_scores.items(), key=lambda x: x[1])
    
    return {
        "best_platform": best_platform[0],
        "platform_scores": platform_scores
    }

def analyze_color_palette(image_pil):
    """Analyze if image uses brand colors using PIL only"""
    
    # Convert PIL to numpy array
    image = np.array(image_pil)
    
    # Resize for faster processing using PIL
    if image_pil.width > 300:
        scale = 300 / image_pil.width
        new_size = (int(image_pil.width * scale), int(image_pil.height * scale))
        image_pil_resized = image_pil.resize(new_size, Image.Resampling.LANCZOS)
        image = np.array(image_pil_resized)
    
    # Extract dominant colors
    pixels = image.reshape(-1, 3)
    
    # Sample pixels if too many (for performance)
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
    
    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    kmeans.fit(pixels)
    dominant_colors = kmeans.cluster_centers_
    
    # Calculate percentages
    labels = kmeans.labels_
    percentages = []
    for i in range(5):
        percentages.append(np.sum(labels == i) / len(labels))
    
    # Sort by percentage
    sorted_idx = np.argsort(percentages)[::-1]
    dominant_colors = dominant_colors[sorted_idx]
    percentages = [percentages[i] for i in sorted_idx]
    
    # Check brand color matches
    brand_matches = []
    for dom_color in dominant_colors:
        for brand_name, brand_color in BRAND_COLORS.items():
            distance = np.linalg.norm(dom_color - brand_color)
            if distance < 60:  # Threshold for "close enough"
                brand_matches.append(brand_name)
                break
    
    return {
        "dominant_colors": dominant_colors,
        "percentages": percentages,
        "brand_matches": brand_matches,
        "brand_color_usage": len(set(brand_matches)) / len(BRAND_COLORS)
    }

def calculate_overall_quality_score(brand_score, audience_score, color_score, platform_scores):
    """Calculate final quality score using simulation formula"""
    
    # Weight components (similar to your simulation)
    weights = {
        "brand_consistency": 0.35,
        "audience_targeting": 0.30,
        "color_palette": 0.20,
        "platform_optimization": 0.15
    }
    
    # Get best platform score
    best_platform_score = max(platform_scores.values())
    
    # Calculate weighted score
    quality_score = (
        brand_score * weights["brand_consistency"] +
        audience_score * weights["audience_targeting"] +
        color_score * weights["color_palette"] +
        best_platform_score * weights["platform_optimization"]
    ) * 100
    
    # Calculate quality factor (matching your simulation)
    if quality_score <= 75:
        quality_factor = 0.7 + (quality_score - 50) * 0.012
    else:
        quality_factor = 1.0 + (quality_score - 75) * 0.02
    
    quality_factor = max(0.7, min(1.5, quality_factor))
    
    return quality_score, quality_factor

# Streamlit App
st.set_page_config(
    page_title="Aura Pet Care - Image Quality Analyzer",
    page_icon="🐾",
    layout="wide"
)

st.title("🐾 Aura Pet Care - Social Media Image Quality Analyzer")
st.markdown("---")

# Sidebar with guidelines
with st.sidebar:
    st.header("📋 What Makes a Good Image?")
    
    st.subheader("🎨 Brand Consistency")
    st.write("""
    - **Colors**: Soft pastels (lavender, sage, cream)
    - **Mood**: Calming, wellness-focused
    - **Quality**: Professional, premium feel
    - **Theme**: Natural pet wellness
    """)
    
    st.subheader("🎯 Audience Targeting")
    st.write("""
    **Youth (18-24)**
    - Authentic, shareable moments
    - Bright, energetic visuals
    - UGC style preferred
    
    **Core (25-44)**
    - Professional yet approachable
    - Family-oriented content
    - Clear product benefits
    
    **Premium (45+)**
    - Sophisticated imagery
    - Expert endorsement visible
    - Traditional quality cues
    """)
    
    st.subheader("📱 Platform Best Practices")
    st.write("""
    **Instagram**: Square/vertical, aesthetic
    **TikTok**: Vertical, dynamic, authentic
    **Facebook**: Clear messaging, informative
    **Pinterest**: Vertical, inspirational
    """)
    
    st.subheader("📊 Quality Score Impact")
    st.write("""
    - **50-75**: 0.7x-1.0x multiplier
    - **75-90**: 1.0x-1.3x multiplier  
    - **90-100**: 1.3x-1.5x multiplier
    """)

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    
    # Target audience selector
    age_group = st.selectbox(
        "Select Target Audience",
        ["18-24 (Youth)", "25-44 (Core)", "45+ (Premium)"]
    )
    
    # Image uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['png', 'jpg', 'jpeg'],
        help="Upload your social media image for quality analysis"
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Analyze button
        if st.button("🔍 Analyze Image Quality", type="primary"):
            with st.spinner("Analyzing image..."):
                # Load model
                model, processor = load_clip_model()
                
                # Run analyses
                brand_analysis = analyze_brand_consistency(image, model, processor)
                audience_analysis = analyze_audience_targeting(image, model, processor, age_group)
                platform_analysis = analyze_platform_suitability(image, model, processor)
                color_analysis = analyze_color_palette(image)
                
                # Store in session state
                st.session_state['analysis_complete'] = True
                st.session_state['brand_analysis'] = brand_analysis
                st.session_state['audience_analysis'] = audience_analysis
                st.session_state['platform_analysis'] = platform_analysis
                st.session_state['color_analysis'] = color_analysis

# Results column
with col2:
    if st.session_state.get('analysis_complete', False):
        st.header("📊 Analysis Results")
        
        # Calculate overall score
        quality_score, quality_factor = calculate_overall_quality_score(
            st.session_state['brand_analysis']['overall_score'],
            st.session_state['audience_analysis']['audience_fit_score'],
            st.session_state['color_analysis']['brand_color_usage'],
            st.session_state['platform_analysis']['platform_scores']
        )
        
        # Overall score display
        st.metric(
            "Overall Quality Score", 
            f"{quality_score:.1f}/100",
            f"Quality Factor: {quality_factor:.2f}x"
        )
        
        # Progress bar
        st.progress(quality_score/100)
        
        # Score breakdown
        st.subheader("📈 Score Breakdown")
        
        col_scores1, col_scores2 = st.columns(2)
        
        with col_scores1:
            st.metric(
                "Brand Consistency",
                f"{st.session_state['brand_analysis']['overall_score']*100:.1f}%"
            )
            st.metric(
                "Audience Targeting",
                f"{st.session_state['audience_analysis']['audience_fit_score']*100:.1f}%"
            )
        
        with col_scores2:
            st.metric(
                "Color Palette Match",
                f"{st.session_state['color_analysis']['brand_color_usage']*100:.1f}%"
            )
            st.metric(
                "Best Platform",
                st.session_state['platform_analysis']['best_platform']
            )
        
        # Detailed insights
        st.subheader("🔍 Detailed Insights")
        
        # Brand consistency details
        with st.expander("Brand Consistency Analysis"):
            st.write("**Top Brand Matches:**")
            for desc, score in st.session_state['brand_analysis']['top_matches']:
                st.write(f"- {desc}: {score*100:.1f}%")
        
        # Audience targeting details
        with st.expander("Audience Targeting Analysis"):
            st.write(f"**Best Match:** {st.session_state['audience_analysis']['best_match']}")
            st.write(f"**Confidence:** {st.session_state['audience_analysis']['match_confidence']*100:.1f}%")
        
        # Platform recommendations
        with st.expander("Platform Recommendations"):
            platform_df = pd.DataFrame(
                list(st.session_state['platform_analysis']['platform_scores'].items()),
                columns=['Platform', 'Suitability']
            )
            platform_df['Suitability'] = platform_df['Suitability'] * 100
            platform_df = platform_df.sort_values('Suitability', ascending=False)
            
            fig = px.bar(
                platform_df, 
                x='Platform', 
                y='Suitability',
                title="Platform Suitability Scores (%)",
                color='Suitability',
                color_continuous_scale='viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Color analysis
        with st.expander("Color Palette Analysis"):
            colors = st.session_state['color_analysis']['dominant_colors']
            percentages = st.session_state['color_analysis']['percentages']
            
            # Create color swatches
            color_data = []
            for i, (color, pct) in enumerate(zip(colors, percentages)):
                color_data.append({
                    'Color': f'Color {i+1}',
                    'Percentage': pct * 100,
                    'RGB': f'rgb({int(color[0])},{int(color[1])},{int(color[2])})'
                })
            
            # Display color swatches
            cols = st.columns(5)
            for i, data in enumerate(color_data):
                with cols[i]:
                    st.markdown(
                        f"""
                        <div style="background-color: {data['RGB']}; 
                                    width: 100%; 
                                    height: 50px; 
                                    border-radius: 5px;
                                    margin-bottom: 5px;">
                        </div>
                        <p style="text-align: center; font-size: 12px;">
                            {data['Percentage']:.1f}%
                        </p>
                        """,
                        unsafe_allow_html=True
                    )
            
            if st.session_state['color_analysis']['brand_matches']:
                st.success(f"✅ Found brand colors: {', '.join(set(st.session_state['color_analysis']['brand_matches']))}")
            else:
                st.warning("⚠️ No brand colors detected")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if quality_score < 75:
            st.error("❌ Image needs improvement for optimal performance")
            
            if st.session_state['brand_analysis']['overall_score'] < 0.6:
                st.write("- 📸 Ensure image reflects Aura's calming, premium pet wellness brand")
            
            if st.session_state['audience_analysis']['audience_fit_score'] < 0.6:
                st.write(f"- 🎯 Adjust content style to better match {age_group} preferences")
            
            if st.session_state['color_analysis']['brand_color_usage'] < 0.4:
                st.write("- 🎨 Incorporate more brand colors (lavender, sage, cream)")
        
        elif quality_score < 90:
            st.warning("⚠️ Good image with room for optimization")
            st.write("- Consider minor adjustments to reach 90+ score")
            st.write("- Test on recommended platform:", st.session_state['platform_analysis']['best_platform'])
        
        else:
            st.success("✅ Excellent image quality!")
            st.write("- Ready for posting on", st.session_state['platform_analysis']['best_platform'])
            st.write("- Expected performance boost: {:.0f}%".format((quality_factor - 1) * 100))

# Footer
st.markdown("---")
st.caption("🐾 Aura Cosmetics - Transforming Pet Wellness Through Natural Care")