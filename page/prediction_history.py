"""
Prediction History Page - View all past predictions
"""
import streamlit as st
from utils.db_handler import get_user_prediction_history, get_prediction_stats
from datetime import datetime


def format_datetime(dt):
    """Format datetime for display"""
    if isinstance(dt, str):
        try:
            dt = datetime.fromisoformat(dt)
        except:
            return dt
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "N/A"


def prediction_history_page(cookies=None):
    """
    Display user's prediction history with filtering and statistics
    """
    st.set_page_config(page_title="Prediction History", page_icon="📜", layout="wide")
    st.title("📜 Prediction History")
    
    # Check if user is logged in
    if 'user_id' not in st.session_state:
        st.error("⚠️ Please login first to view your prediction history")
        if st.button("Go to Login"):
            st.session_state['page'] = "login"
            st.session_state['authenticated'] = False
            st.rerun()
        return
    
    user_id = st.session_state['user_id']
    
    # Get statistics
    stats = get_prediction_stats(user_id)
    
    # Display statistics
    st.subheader("📊 Your Prediction Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Predictions", stats['total_predictions'])
    
    with col2:
        if stats['most_predicted_crop']:
            st.metric("Most Predicted Crop", stats['most_predicted_crop'], 
                     delta=f"{stats['most_predicted_count']} times")
        else:
            st.metric("Most Predicted Crop", "N/A")
    
    with col3:
        st.metric("Recent (7 days)", stats['recent_predictions'])
    
    st.markdown("---")
    
    # Get prediction history
    history = get_user_prediction_history(user_id)
    
    if not history:
        st.info("📭 No prediction history found. Make your first prediction to see it here!")
        if st.button("🌾 Go to Prediction Page"):
            st.session_state['page'] = "PredictPage"
            st.rerun()
        return
    
    # Filter options
    st.subheader("🔍 Filter Predictions")
    col1, col2 = st.columns(2)
    
    with col1:
        filter_type = st.selectbox(
            "Prediction Type",
            ["All", "Normal", "Advanced"],
            key="filter_type"
        )
    
    with col2:
        # Get unique crops from history
        unique_crops = list(set([h['predicted_crop'] for h in history if h['predicted_crop']]))
        unique_crops.sort()
        filter_crop = st.selectbox(
            "Crop",
            ["All"] + unique_crops,
            key="filter_crop"
        )
    
    # Apply filters
    filtered_history = history
    if filter_type != "All":
        filtered_history = [h for h in filtered_history if h['prediction_type'].lower() == filter_type.lower()]
    if filter_crop != "All":
        filtered_history = [h for h in filtered_history if h['predicted_crop'] == filter_crop]
    
    st.markdown("---")
    st.subheader(f"📋 Prediction Records ({len(filtered_history)} results)")
    
    # Display predictions
    for idx, prediction in enumerate(filtered_history, 1):
        with st.expander(
            f"#{idx} - {prediction['predicted_crop']} ({prediction['prediction_type'].upper()}) - {format_datetime(prediction['created_at'])}",
            expanded=(idx == 1)
        ):
            # Basic info
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown("**📍 Location**")
                st.write(f"District: {prediction['district']}")
                st.write(f"Taluka: {prediction['taluka']}")
            
            with col2:
                st.markdown("**🌱 Soil Info**")
                st.write(f"Type: {prediction['soil_type']}")
                st.write(f"pH: {prediction['soil_ph']:.2f}")
            
            with col3:
                st.markdown("**🎯 Prediction**")
                st.write(f"Crop: {prediction['predicted_crop']}")
                if prediction['confidence']:
                    st.write(f"Confidence: {prediction['confidence']:.2f}%")
            
            with col4:
                st.markdown("**📈 Yield**")
                if prediction['predicted_yield']:
                    st.write(f"{prediction['predicted_yield']:.2f} Q/Ha")
                else:
                    st.write("N/A")
                st.write(f"Date: {format_datetime(prediction['created_at'])}")
            
            st.markdown("---")
            
            # Detailed results
            result_data = prediction['prediction_result']
            
            if prediction['prediction_type'] == 'normal':
                st.markdown("### 🏆 Top Recommendations")
                
                recommendations = result_data.get('top_recommendations', [])
                for i, rec in enumerate(recommendations, 1):
                    col1, col2, col3, col4 = st.columns(4)
                    col1.write(f"**{i}. {rec['crop']}**")
                    col2.write(f"Probability: {rec['probability']:.2f}%")
                    col3.write(f"Yield: {rec['predicted_yield_quintal_per_ha']:.2f} Q/Ha")
                    col4.write(f"Suitability: {rec['suitability']}")
            
            elif prediction['prediction_type'] == 'advanced':
                st.markdown("### 🎯 Detailed Analysis")
                
                col1, col2 = st.columns(2)
                with col1:
                    prediction_label = result_data.get('prediction', 'Unknown')
                    if prediction_label == "Grow":
                        st.success(f"✅ **{prediction_label}** - This crop CAN grow in your region")
                    else:
                        st.error(f"❌ **{prediction_label}** - This crop may NOT grow well")
                
                with col2:
                    st.info(f"**Confidence:** {result_data.get('confidence', 0):.2f}%")
                
                # Reason
                reason = result_data.get('reason', '')
                if reason:
                    st.markdown("**💡 Analysis:**")
                    st.info(reason)
                
                # Treatment
                treatment = result_data.get('treatment')
                if treatment:
                    st.markdown("**📝 Treatment Plan:**")
                    
                    treat_col1, treat_col2 = st.columns(2)
                    
                    with treat_col1:
                        if 'pestManagement' in treatment:
                            st.write(f"**Pest Management:** {treatment['pestManagement']}")
                        if 'recommendedIrrigation' in treatment:
                            st.write(f"**Irrigation:** {treatment['recommendedIrrigation']}")
                    
                    with treat_col2:
                        if 'recommendedFertilizers' in treatment:
                            st.write(f"**Fertilizers:** {', '.join(treatment['recommendedFertilizers'])}")
                        if 'recommendedPesticides' in treatment:
                            st.write(f"**Pesticides:** {', '.join(treatment['recommendedPesticides'])}")
            
            # Raw JSON view
            with st.expander("🔍 View Raw JSON"):
                st.json(result_data)
    
    # Navigation
    st.markdown("---")
    nav_cols = st.columns(3)
    
    with nav_cols[0]:
        if st.button("⬅️ Back to Home", use_container_width=True):
            st.session_state['page'] = "app"
            st.rerun()
    
    with nav_cols[1]:
        if st.button("🌾 New Prediction", use_container_width=True):
            st.session_state['page'] = "PredictPage"
            st.rerun()
    
    with nav_cols[2]:
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()


if __name__ == "__main__":
    prediction_history_page()
