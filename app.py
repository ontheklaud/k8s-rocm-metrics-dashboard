import os

from datetime import datetime
import json
import time

import streamlit as st
import requests
import plotly.graph_objects as go
import pandas as pd


# Configure the page
st.set_page_config(
    page_title="GPU Metrics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Constants
PROMETHEUS_METRICS_ENDPOINT = os.environ.get("PROMETHEUS_METRICS_ENDPOINT", "http://localhost:9090/api/v1/query")
PROMETHEUS_METRICS_PODNAME = os.environ.get("PROMETHEUS_METRICS_PODNAME", "prometheus")
REFRESH_INTERVAL = 5  # seconds

GPU_NAME_RESOLVE = {
    "102-D65209-00": "MI250",
    "102-G30211-0C": "MI300",
    "102-G30219-00": "MI308X",
}

# GPU model power limits (in watts)
GPU_POWER_LIMITS = {
    'MI250': 560,  # AMD Instinct MI250
    "MI300": 750,  # AMD Instinct MI300
    "MI308X": 650,  # AMD Instinct MI308X
    'default': 300  # Default power limit for unknown models
}

# Color scheme for gauges
GAUGE_COLORS = {
    'green': '#2ecc71',      # 0-20% (value bar)
    'light_green': '#27ae60', # 20-40% (value bar)
    'yellow': '#f1c40f',     # 40-60% (value bar)
    'orange': '#e67e22',     # 60-80% (value bar)
    'red': '#e74c3c',        # 80-100% (value bar)

    # Less saturated colors for gauge plates
    'plate_green': '#a8e6cf',      # 0-20%
    'plate_light_green': '#88d8b0', # 20-40%
    'plate_yellow': '#ffd3b6',     # 40-60%
    'plate_orange': '#ffaaa5',     # 60-80%
    'plate_red': '#ff8b94'         # 80-100%
}

def get_color_for_value(value, max_val):
    """Get color based on value percentage"""
    percentage = (value / max_val) * 100
    if percentage <= 20:
        return GAUGE_COLORS['green']
    elif percentage <= 40:
        return GAUGE_COLORS['light_green']
    elif percentage <= 60:
        return GAUGE_COLORS['yellow']
    elif percentage <= 80:
        return GAUGE_COLORS['orange']
    else:
        return GAUGE_COLORS['red']

def create_gauge(value, title, min_val=0, max_val=100, height=400):
    """Create a gauge chart using Plotly with color scheme"""
    color = get_color_for_value(value, max_val)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        gauge={
            'axis': {
                'range': [min_val, max_val],
                'tickmode': 'linear',
                'tick0': min_val,
                'dtick': max_val/5,
                'showticklabels': True
            },
            'bar': {
                'color': color,
                'line': {'color': 'black', 'width': 1}
            },
            'steps': [
                {'range': [0, max_val*0.2], 'color': GAUGE_COLORS['plate_green']},
                {'range': [max_val*0.2, max_val*0.4], 'color': GAUGE_COLORS['plate_light_green']},
                {'range': [max_val*0.4, max_val*0.6], 'color': GAUGE_COLORS['plate_yellow']},
                {'range': [max_val*0.6, max_val*0.8], 'color': GAUGE_COLORS['plate_orange']},
                {'range': [max_val*0.8, max_val], 'color': GAUGE_COLORS['plate_red']}
            ]
        }
    ))
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=30, t=0, b=0)  # Adjust margins to make gauge smaller
    )
    return fig

def create_horizontal_bar(value, title, min_val=0, max_val=100, height=400):
    """Create a horizontal bar chart using Plotly with color scheme"""
    color = get_color_for_value(value, max_val)
    
    fig = go.Figure(go.Bar(
        x=[value],
        y=[title],
        orientation='h',
        marker_color=color,
        marker_line_color='gray',
        marker_line_width=2,
        width=0.5
    ))
    
    fig.update_layout(
        xaxis=dict(
            range=[min_val, max_val],
            showgrid=True,
            gridcolor='lightgray'
        ),
        yaxis=dict(showticklabels=False),
        height=height,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    
    # Add background color steps
    for i, (start, end, color) in enumerate([
        (0, max_val*0.2, GAUGE_COLORS['plate_green']),
        (max_val*0.2, max_val*0.4, GAUGE_COLORS['plate_light_green']),
        (max_val*0.4, max_val*0.6, GAUGE_COLORS['plate_yellow']),
        (max_val*0.6, max_val*0.8, GAUGE_COLORS['plate_orange']),
        (max_val*0.8, max_val, GAUGE_COLORS['plate_red'])
    ]):
        fig.add_shape(
            type="rect",
            x0=start,
            x1=end,
            y0=-0.5,
            y1=0.5,
            fillcolor=color,
            opacity=0.3,
            layer="below",
            line_width=0
        )
    
    return fig

def fetch_gpu_metrics():
    """Fetch GPU metrics from Prometheus and return as DataFrame"""
    try:
        # 1. Get node IP of the running node
        param_get_physical_node = f"kube_pod_info{{pod=~\".*{PROMETHEUS_METRICS_PODNAME}.*\"}}"
        resp_get_physical_node = requests.get(
            url=PROMETHEUS_METRICS_ENDPOINT,
            params={"query": param_get_physical_node}
        )
        resp_get_physical_node.raise_for_status()
        dict_info_physical_node = json.loads(resp_get_physical_node.text)
        executor_node_ip = dict_info_physical_node["data"]["result"][0]["metric"]["host_ip"]

        # 2. Get all GPU metrics
        promql_get_gpu_usage = (
            "{__name__=~\""
            "amd_gpu_edge_temperature|amd_gpu_gfx_activity|amd_gpu_average_package_power|"
            "amd_gpu_used_vram|amd_gpu_total_vram\", instance=~\"" + 
            executor_node_ip + ":.+\"}"
        )
        resp_get_gpu_usage = requests.get(
            url=PROMETHEUS_METRICS_ENDPOINT,
            params={"query": promql_get_gpu_usage}
        )
        resp_get_gpu_usage.raise_for_status()
        dict_info_gpu_usage = json.loads(resp_get_gpu_usage.text)

        dict_gpu_model = {}
        # 3. Process metrics into DataFrame
        metrics_data = []
        for metric in dict_info_gpu_usage["data"]["result"]:
            metric_inst = metric["metric"]
            metrics_data.append({
                'gpu_id': metric_inst["gpu_id"],
                'metric_name': metric_inst["__name__"],
                'value': float(metric["value"][1]),
            })

            if metric_inst["gpu_id"] not in dict_gpu_model:
                dict_gpu_model.update({metric_inst["gpu_id"]: metric_inst["card_model"]})
            else:
                continue

        for gpu_id, card_model in dict_gpu_model.items():
            metrics_data.append({
                'gpu_id': gpu_id,
                'metric_name': "card_model",
                'value': card_model
            })
        
        # Create DataFrame
        df = pd.DataFrame(metrics_data)
        
        # Pivot the DataFrame to get metrics as columns
        df_pivot = df.pivot(index='gpu_id', columns='metric_name', values='value')
        
        # Calculate VRAM usage ratio
        df_pivot['vram_usage_ratio'] = (
            df_pivot['amd_gpu_used_vram'] / df_pivot['amd_gpu_total_vram']
        ) * 100

        # Calculate statistics (excluding card_model)
        # numeric_columns = df_pivot.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in df_pivot.columns if col != 'card_model']
        stats = {
            'mean': df_pivot[numeric_columns].mean(),
            'max': df_pivot[numeric_columns].max(),
            'min': df_pivot[numeric_columns].min()
        }
        
        return df_pivot, stats

    except Exception as e:
        st.error(f"Error fetching GPU metrics: {str(e)}")
        return None, None

def get_power_limit(card_model):
    """Get power limit for a specific GPU model"""
    resolved_model = GPU_NAME_RESOLVE.get(card_model, card_model)
    return GPU_POWER_LIMITS.get(resolved_model, GPU_POWER_LIMITS['default'])

def create_visualization(value, title, max_val, height, key, gpu_id=None, gpu_metrics_df=None):
    """Create visualization based on selected style with dynamic max values for power"""
    if title.endswith("Power Usage (W)") and gpu_id is not None and gpu_metrics_df is not None:
        # Get power limit based on GPU model
        card_model = gpu_metrics_df.loc[gpu_id, 'card_model']
        resolved_model = GPU_NAME_RESOLVE.get(card_model, card_model)
        max_val = GPU_POWER_LIMITS.get(resolved_model, GPU_POWER_LIMITS['default'])
    
    if st.session_state.use_gauge:
        return create_gauge(value, title, max_val=max_val, height=height)
    else:
        return create_horizontal_bar(value, title, max_val=max_val, height=height)

def main():
    st.title("GPU Metrics Dashboard")
    st.markdown("Real-time monitoring of GPU metrics")

    # Initialize session state for GPU selection and visualization style if they don't exist
    if 'selected_gpus' not in st.session_state:
        st.session_state.selected_gpus = []
    if 'use_gauge' not in st.session_state:
        st.session_state.use_gauge = True

    # Visualization style toggle
    st.header("Display Settings")
    use_gauge = st.toggle("Use Gauge Visualization", value=st.session_state.use_gauge)
    st.session_state.use_gauge = use_gauge

    # Get initial GPU list
    gpu_metrics_df, _ = fetch_gpu_metrics()
    available_gpus = gpu_metrics_df.index.tolist() if gpu_metrics_df is not None else []

    # GPU Selection section - outside the refresh loop
    st.header("GPU Selection")
    num_columns = 4
    gpu_cols = st.columns(num_columns)
    
    # Initialize session state for GPU selection if not exists
    if 'selected_gpus' not in st.session_state:
        st.session_state.selected_gpus = []
    if 'last_selection' not in st.session_state:
        st.session_state.last_selection = None
    
    # Get sorted list of available GPUs
    sorted_gpus = sorted(available_gpus)
    
    # Reset selected GPUs if they're not in available_gpus
    st.session_state.selected_gpus = [gpu for gpu in st.session_state.selected_gpus if gpu in sorted_gpus]
    
    # If no GPUs are selected, select only the first GPU
    if not st.session_state.selected_gpus and sorted_gpus:
        st.session_state.selected_gpus = [sorted_gpus[0]]
    
    # Create checkboxes in a grid
    for i, gpu_id in enumerate(sorted_gpus):
        col_idx = i % num_columns
        with gpu_cols[col_idx]:
            # Create a unique key for each checkbox
            checkbox_key = f"gpu_checkbox_{gpu_id}"
            
            # Get the current state
            is_currently_selected = gpu_id in st.session_state.selected_gpus
            
            # Create the checkbox
            is_selected = st.checkbox(
                f"GPU {gpu_id}",
                value=is_currently_selected,
                key=checkbox_key
            )
            
            # Handle state changes
            if is_selected != is_currently_selected:
                if is_selected:
                    st.session_state.selected_gpus.append(gpu_id)
                else:
                    st.session_state.selected_gpus.remove(gpu_id)
                st.session_state.last_selection = gpu_id
    
    # Sort the selected GPUs list
    st.session_state.selected_gpus.sort()
    
    # Debug information
    st.sidebar.write("Debug Info:")
    st.sidebar.write("Current selections:", st.session_state.selected_gpus)
    st.sidebar.write("Last selection:", st.session_state.last_selection)

    # Create placeholder for auto-refresh
    placeholder = st.empty()

    height_average = 300
    height_specific = 200

    while True:
        current_time = datetime.now()
        timestamp = current_time.strftime("%Y%m%d%H%M%S%f")
        
        with placeholder.container():
            gpu_metrics_df, stats = fetch_gpu_metrics()
            
            if gpu_metrics_df is not None:
                # Filter selected GPUs
                filtered_df = gpu_metrics_df.loc[st.session_state.selected_gpus]
                
                # Calculate averages for selected GPUs (excluding card_model)
                numeric_columns = [col for col in filtered_df.columns if col != 'card_model']
                averages = filtered_df[numeric_columns].mean()
                
                # Special handling for power usage average - ignore zero values
                power_values = filtered_df['amd_gpu_average_package_power']
                non_zero_power = power_values[power_values > 0]
                if not non_zero_power.empty:
                    averages['amd_gpu_average_package_power'] = non_zero_power.mean()
                
                # Display average metrics
                st.subheader("Average Metrics (Selected GPUs)")
                avg_cols = st.columns(4)
                
                with avg_cols[0]:
                    st.plotly_chart(
                        create_visualization(
                            averages.get("amd_gpu_gfx_activity", 0),
                            "Avg GPU Utilization (%)",
                            max_val=100,
                            height=height_average,
                            key=f"avg_gpu_util_{timestamp}",
                            gpu_id=st.session_state.selected_gpus[0] if st.session_state.selected_gpus else None,
                            gpu_metrics_df=gpu_metrics_df
                        ),
                        use_container_width=True,
                        key=f"plot_avg_gpu_util_{timestamp}"
                    )
                
                with avg_cols[1]:
                    st.plotly_chart(
                        create_visualization(
                            averages.get("vram_usage_ratio", 0),
                            "Avg VRAM Usage (%)",
                            max_val=100,
                            height=height_average,
                            key=f"avg_vram_usage_{timestamp}",
                            gpu_id=None,
                            gpu_metrics_df=None
                        ),
                        use_container_width=True,
                        key=f"plot_avg_vram_usage_{timestamp}"
                    )
                
                with avg_cols[2]:
                    st.plotly_chart(
                        create_visualization(
                            averages.get("amd_gpu_edge_temperature", 0),
                            "Avg Temperature (°C)",
                            max_val=100,
                            height=height_average,
                            key=f"avg_temp_{timestamp}",
                            gpu_id=None,
                            gpu_metrics_df=None
                        ),
                        use_container_width=True,
                        key=f"plot_avg_temp_{timestamp}"
                    )
                
                with avg_cols[3]:
                    st.plotly_chart(
                        create_visualization(
                            averages.get("amd_gpu_average_package_power", 0),
                            "Avg Power Usage (W)",
                            max_val=300,
                            height=height_average,
                            key=f"avg_power_{timestamp}",
                            gpu_id=st.session_state.selected_gpus[0] if st.session_state.selected_gpus else None,
                            gpu_metrics_df=gpu_metrics_df
                        ),
                        use_container_width=True,
                        key=f"plot_avg_power_{timestamp}"
                    )

                # Display individual GPU metrics
                st.subheader("Individual GPU Metrics")
                for gpu_id, metrics in filtered_df.iterrows():
                    card_model = metrics.get('card_model', 'Unknown')
                    st.markdown(f"### GPU {gpu_id} ({GPU_NAME_RESOLVE.get(card_model)})")
                    gpu_cols = st.columns(4)
                    
                    with gpu_cols[0]:
                        st.plotly_chart(
                            create_visualization(
                                metrics.get("amd_gpu_gfx_activity", 0),
                                "GPU Utilization (%)",
                                max_val=100,
                                height=height_specific,
                                key=f"gpu_util_{gpu_id}_{timestamp}",
                                gpu_id=gpu_id,
                                gpu_metrics_df=gpu_metrics_df
                            ),
                            use_container_width=True,
                            key=f"plot_gpu_util_{gpu_id}_{timestamp}"
                        )
                    
                    with gpu_cols[1]:
                        st.plotly_chart(
                            create_visualization(
                                metrics.get("vram_usage_ratio", 0),
                                "VRAM Usage (%)",
                                max_val=100,
                                height=height_specific,
                                key=f"vram_usage_{gpu_id}_{timestamp}",
                                gpu_id=gpu_id,
                                gpu_metrics_df=gpu_metrics_df
                            ),
                            use_container_width=True,
                            key=f"plot_vram_usage_{gpu_id}_{timestamp}"
                        )
                    
                    with gpu_cols[2]:
                        st.plotly_chart(
                            create_visualization(
                                metrics.get("amd_gpu_edge_temperature", 0),
                                "Temperature (°C)",
                                max_val=100,
                                height=height_specific,
                                key=f"temp_{gpu_id}_{timestamp}",
                                gpu_id=gpu_id,
                                gpu_metrics_df=gpu_metrics_df
                            ),
                            use_container_width=True,
                            key=f"plot_temp_{gpu_id}_{timestamp}"
                        )
                    
                    with gpu_cols[3]:
                        st.plotly_chart(
                            create_visualization(
                                metrics.get("amd_gpu_average_package_power", 0),
                                "Power Usage (W)",
                                max_val=300,
                                height=height_specific,
                                key=f"power_{gpu_id}_{timestamp}",
                                gpu_id=gpu_id,
                                gpu_metrics_df=gpu_metrics_df
                            ),
                            use_container_width=True,
                            key=f"plot_power_{gpu_id}_{timestamp}"
                        )

                # Display statistics
                st.subheader("GPU Metrics Statistics")
                stats_df = pd.DataFrame(stats).round(2)
                st.dataframe(stats_df, use_container_width=True)

                # Last updated timestamp
                st.text(f"Last updated: {current_time.strftime('%Y-%m-%d %H:%M:%S')}")

        time.sleep(REFRESH_INTERVAL)

if __name__ == "__main__":
    main() 