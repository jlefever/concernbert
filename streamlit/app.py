import sys
sys.path.append("src")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.stats import johnsonsu
from sklearn.manifold import MDS

from concernbert.semantic import find_entity_docs, EntityDoc
from concernbert.frontend import CdCalculator, JOHNSONSU_PARAMS
from concernbert.frontend import estimate_percentile as ep

def to_dfs(groups: list[list[EntityDoc]]) -> list[pd.DataFrame]:
    dfs: list[pd.DataFrame] = []
    for group in groups:
        records = [{"name": d.name, "kind": d.kind, "lineno": d.lineno} for d in group]
        dfs.append(pd.DataFrame.from_records(records))
    return dfs


def plot_pdf(highlight_values=[]) -> None:
    shape1, shape2, loc, scale = JOHNSONSU_PARAMS
    x_min = johnsonsu.ppf(0.001, shape1, shape2, loc, scale)
    x_max = johnsonsu.ppf(0.999, shape1, shape2, loc, scale)
    x = np.linspace(x_min, x_max, 10000)
    pdf = johnsonsu.pdf(x, shape1, shape2, loc, scale)
    
    # Create interactive figure using Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=pdf, mode='lines', name='PDF', line=dict(color='blue')))
    
    # Highlight specific points
    if highlight_values:
        highlight_pdf = johnsonsu.pdf(highlight_values, shape1, shape2, loc, scale)
        fig.add_trace(go.Scatter(
            x=highlight_values,
            y=highlight_pdf,
            mode='markers',
            marker=dict(color='red', size=8),
            name="Groups"
        ))
        for val, pdf_val in zip(highlight_values, highlight_pdf):
            fig.add_trace(go.Scatter(
                x=[val, val], y=[0, pdf_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                showlegend=False
            ))
    
    # Labels and Title
    fig.update_layout(
        title='CD Density',
        xaxis_title='CD',
        yaxis_title='Density',
        template='ggplot2',
        xaxis=dict(range=[x_min, x_max]),
        yaxis=dict(range=[0, max(pdf) * 1.1]),
    )
    
    # Display plot in Streamlit
    st.plotly_chart(fig)


def plot_embeddings(seed, embeddings, names, true_dist=False):
    if len(embeddings) != len(names):
        st.error("Embeddings and names lists must be the same length.")
        return
    
    if true_dist:
        # Compute mean in original space and distances before projection
        mean_embedding = np.mean(embeddings, axis=0)
        distances = np.linalg.norm(embeddings - mean_embedding, axis=1)
        avg_distance = np.mean(distances)
    
    # Project embeddings to 2D using MDS
    mds = MDS(n_components=2, random_state=seed)
    projected_embeddings = mds.fit_transform(embeddings)
    
    if not true_dist:
        # Compute mean in projected space
        mean_embedding = np.mean(projected_embeddings, axis=0)
        distances = np.linalg.norm(projected_embeddings - mean_embedding, axis=1)
        avg_distance = np.mean(distances)
    
    # Create plotly scatter plot
    fig = go.Figure()
    
    # Plot embeddings (same color) with labels always displayed
    fig.add_trace(go.Scatter(
        x=projected_embeddings[:, 0],
        y=projected_embeddings[:, 1],
        mode='markers+text',
        marker=dict(size=8, color='blue'),
        text=names,
        textposition='top center',
        name='Embeddings'
    ))
    
    # Plot mean
    fig.add_trace(go.Scatter(
        x=[mean_embedding[0]], y=[mean_embedding[1]],
        mode='markers+text',
        marker=dict(size=10, color='red'),
        text='Mean (2D)',
        textposition='top center',
        name='Mean (2D)'
    ))
    
    # Draw dotted circle centered at mean
    circle_theta = np.linspace(0, 2*np.pi, 1000)
    circle_x = mean_embedding[0] + avg_distance * np.cos(circle_theta)
    circle_y = mean_embedding[1] + avg_distance * np.sin(circle_theta)
    fig.add_trace(go.Scatter(
        x=circle_x, y=circle_y,
        mode='lines',
        line=dict(color='black', dash='dot'),
        name='Concern Deviation (2D)'
    ))
    
    # Labels and layout with equal aspect ratio and unit grid lines
    fig.update_layout(
        title='MDS Projection of Embeddings',
        xaxis_title='X',
        yaxis_title='Y',
        template='plotly_white',
        xaxis=dict(showgrid=True, zeroline=True, scaleanchor='y', dtick=0.25),
        yaxis=dict(showgrid=True, zeroline=True, scaleanchor='x', dtick=0.25)
    )
    
    # Display plot in Streamlit
    st.plotly_chart(fig)



cd_calculator = CdCalculator("_models/EntityBERT-v3_train_nonldl-lr5e5-2_83-e3/", "_cache/")

# st.set_page_config(layout="wide")
st.title("ConcernBERT")

st.markdown(
    """
    <style>
        .stMainBlockContainer {
            max-width: 1100px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("Enter the contents of a Java source file below.")

default_source = """import java.util.ArrayList;
import java.util.List;

public class Student {
    private final String name;
    private final List<Course> courses;

    public Student(String name) {
        this.name = name;
        this.courses = new ArrayList<>();
    }

    public String getName() {
        return name;
    }

    public List<Course> getCourses() {
        return courses;
    }

    public float getTermGpa() {
        if (courses.isEmpty()) {
            return 0.0f;
        }
        return getTermGradePoints() / (getTermCredits() == 0 ? 1 : getTermCredits());
    }

    public int getTermCredits() {
        return courses.stream().mapToInt(Course::getCredits).sum();
    }

    public float getTermGradePoints() {
        return (float) courses.stream()
            .mapToDouble(course -> course.getGrade() * course.getCredits())
            .sum();
    }

    public void addCourse(int credits, float grade) {
        courses.add(new Course(credits, grade));
    }

    private static class Course {
        private final int credits;
        private final float grade;

        public Course(int credits, float grade) {
            this.credits = credits;
            this.grade = grade;
        }

        public int getCredits() {
            return credits;
        }

        public float getGrade() {
            return grade;
        }
    }
}"""

with st.expander("Source Code (Edit)", expanded=True):
    source = st.text_area("Source", value=default_source, height=300, label_visibility="collapsed")

with st.expander("Source Code (View)", expanded=True):
    st.code(source, language="java", line_numbers=True, wrap_lines=False)

groups = find_entity_docs(source)

if len(groups) == 0:
    st.warning("No entities found in source")
    st.stop()

dfs = to_dfs(groups)
multi_group = len(groups) > 1

tabs = st.tabs([f"Group {i + 1}" for i in range(len(groups))])
edited_groups: list[list[EntityDoc]] = []

for i, tab in enumerate(tabs):
    with tab:
        df = dfs[i].copy()
        columns = list(df.columns)
        df.insert(0, "Enabled", [True] * len(df))
        edited_df = st.data_editor(df, disabled=columns, use_container_width=True, key=f"editable_table_{i}")
        enabled = list(edited_df["Enabled"])
        edited_group = [d for j, d in enumerate(groups[i]) if enabled[j]]
        edited_groups.append(edited_group)
        cd_result = cd_calculator.calc_cd_from_docs([edited_group])
        st.markdown(f"**CD of Group {i + 1}:** {cd_result.groups[0]:.4f} (Greater than ≈{ep(cd_result.groups[0]):.2f}%)")
        seed = st.number_input("Seed", key=f"seed-{i}", value=42, min_value=0)
        # true_dist = st.checkbox("True Distances", key=f"true-dist-{i}")
        names = [d.name for d in edited_group]
        plot_embeddings(seed, cd_result.embeddings[0], names, true_dist=False)

st.divider()

cd_result = cd_calculator.calc_cd_from_docs(edited_groups)

if multi_group:
    st.markdown(f"**Average CD:** {cd_result.inter_cd:.4f} (Greater than ≈{ep(cd_result.inter_cd):.2f}%)" , help="Calculate the CD of each group, then average.")
else:
    st.markdown(f"**CD:** {cd_result.inter_cd:.4f} (Greater than ≈{ep(cd_result.inter_cd):.2f}%)")


plot_pdf(highlight_values=cd_result.groups)