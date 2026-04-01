# --coding=utf-8--
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

PRICE_Path = './anymous/name_price_anonymized.xlsx'
PIC_Des = './result_kg_reproduce/price_distribution_pic/'


def generate_academic_figures(file_path, output_dir):
    """
    Generates three high-resolution academic figures:
    1. An individual plot for the original price distribution (visually truncated for clarity).
    2. An individual plot for the natural log-transformed price distribution (Ln).
    3. A combined 1x2 panel plot containing both.
    """
    # 1. Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 2. Read the Excel file directly (Ensure openpyxl is installed)
    try:
        df = pd.read_excel(file_path)
    except Exception as e:
        print(f"Error reading the file. Ensure the path is correct and openpyxl is installed. Details: {e}")
        return

    # 3. Data cleaning: Drop missing values and prices <= 0 to prevent log transformation errors
    df = df.dropna(subset=['price'])
    df = df[df['price'] > 0]

    # 4. Calculate the natural logarithm (base e) of the price
    df['ln_price'] = np.log(df['price'])

    # 5. Calculate the 99th percentile of the original price for visual truncation
    # This prevents the histogram from collapsing into a single vertical line due to extreme outliers.
    p99_price = np.percentile(df['price'], 99)
    df_zoomed = df[df['price'] <= p99_price]

    # ==========================================
    # FIGURE 1: Original Price Individual Plot
    # ==========================================
    plt.figure(figsize=(6, 5))
    sns.histplot(df_zoomed['price'], bins=50, kde=True, color='skyblue', edgecolor='black', alpha=0.7)
    sns.rugplot(df['price'], color='darkblue', alpha=0.5, height=0.03)
    plt.xlim(0, p99_price)
    plt.title('Distribution of Original Price (Truncated at 99th Pct)', fontsize=12, fontweight='bold')
    plt.xlabel('Price', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)

    plt.savefig(os.path.join(output_dir, 'original_price_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # FIGURE 2: Ln(Price) Individual Plot
    # ==========================================
    plt.figure(figsize=(6, 5))
    sns.histplot(df['ln_price'], bins=50, kde=True, color='salmon', edgecolor='black', alpha=0.7)
    sns.rugplot(df['ln_price'], color='darkred', alpha=0.5, height=0.03)
    plt.title('Distribution of Ln(Price)', fontsize=12, fontweight='bold')
    plt.xlabel('Ln(Price)', fontsize=11)
    plt.ylabel('Frequency', fontsize=11)

    plt.savefig(os.path.join(output_dir, 'ln_price_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ==========================================
    # FIGURE 3: Combined 1x2 Panel Plot
    # ==========================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left Panel: Original Price (Truncated)
    sns.histplot(df_zoomed['price'], bins=50, kde=True, ax=axes[0], color='skyblue', edgecolor='black', alpha=0.7)
    sns.rugplot(df['price'], ax=axes[0], color='darkblue', alpha=0.5, height=0.03)
    axes[0].set_xlim(0, p99_price)
    axes[0].set_title('(a) Distribution of Original Price (Truncated at 99th Pct)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Price', fontsize=11)
    axes[0].set_ylabel('Frequency', fontsize=11)

    # Right Panel: Ln Price
    sns.histplot(df['ln_price'], bins=50, kde=True, ax=axes[1], color='salmon', edgecolor='black', alpha=0.7)
    sns.rugplot(df['ln_price'], ax=axes[1], color='darkred', alpha=0.5, height=0.03)
    axes[1].set_title('(b) Distribution of Ln(Price)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Ln(Price)', fontsize=11)
    axes[1].set_ylabel('Frequency', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'combined_price_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    generate_academic_figures(PRICE_Path, PIC_Des)