# DTrade Capital - Features

This repository contains the core behavioral analysis and risk detection features for the DTrade Capital platform.

## Features

### 1. My Behavior Page
- **Description**: Focused on time-series analysis of trade logs.
- **Core Logic**: EVI (Edge, Value, Impact) score calculator from trade logs.
- **Output**: EVI history chart visualizations.

### 2. Pattern Detection
- **Description**: Rule-based detection of negative trading behaviors.
- **Core Logic**: Detects revenge trades, FOMO (Fear Of Missing Out), and size drift.
- **Output**: Flag counts and danger hour heatmaps.

### 3. My Genome Page
- **Description**: Trader classification using behavioral features.
- **Core Logic**: K-Means clustering to categorize traders into specific archetypes.
- **Output**: Archetype labels (requires > 50 trades).

### 4. Behavioral Fitness Score
- **Description**: Multi-metric scoring model for weekly performance.
- **Core Logic**: Weighted metrics formula derived from trade behavior data.
- **Output**: Weekly scorecard and 8-week performance bar charts.

### 5. Weekly Email + Plan Gating
- **Description**: Automated reporting and account level gating.
- **Core Logic**: Summary generation for reports and access flag management.
- **Output**: Monday email reports and free/basic access control.

## Contribution Guidelines
Please upload your code to the corresponding feature folder. Ensure your Python scripts follow the project's coding standards.
"# D-Terminal-features" 
