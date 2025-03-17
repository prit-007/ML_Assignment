import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:frontend/utils/extensions.dart';

import '../models/analysis_result.dart';

class ModelComparisonWidget extends StatelessWidget {
  final Map<String, ModelResult> modelResults;

  ModelComparisonWidget({required this.modelResults});

  @override
  Widget build(BuildContext context) {
    if (modelResults.isEmpty) {
      return Center(child: Text('No model results available for comparison'));
    }

    // Determine if we're dealing with classification or regression
    bool isClassification = false;
    if (modelResults.values.first.metrics.containsKey('accuracy')) {
      isClassification = true;
    }

    // Get the primary metric for sorting
    String primaryMetric = isClassification ? 'accuracy' : 'rmse';

    // Sort models by performance
    var sortedModels = modelResults.entries.toList();
    if (isClassification) {
      // For classification, higher accuracy is better
      sortedModels.sort((a, b) => b.value.metrics[primaryMetric]!
          .compareTo(a.value.metrics[primaryMetric]!));
    } else {
      // For regression, lower RMSE is better
      sortedModels.sort((a, b) => a.value.metrics[primaryMetric]!
          .compareTo(b.value.metrics[primaryMetric]!));
    }

    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildPerformanceComparisonChart(
              context, sortedModels, isClassification, primaryMetric),
          SizedBox(height: 24),
          _buildModelComparisonTable(context, sortedModels, isClassification),
        ],
      ),
    );
  }

  Widget _buildPerformanceComparisonChart(
      BuildContext context,
      List<MapEntry<String, ModelResult>> sortedModels,
      bool isClassification,
      String primaryMetric) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Model Performance Comparison',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 8),
            Text(
              isClassification
                  ? 'Accuracy (higher is better)'
                  : 'Root Mean Squared Error (lower is better)',
              style: TextStyle(color: Colors.grey[600]),
            ),
            SizedBox(height: 16),
            Container(
              height: 300,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: isClassification
                      ? 100 // For percentage accuracy
                      : _getMaxValue(sortedModels, primaryMetric) * 1.2,
                  titlesData: FlTitlesData(
                    leftTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        reservedSize: 40,
                      ),
                    ),
                    bottomTitles: AxisTitles(
                      sideTitles: SideTitles(
                        showTitles: true,
                        getTitlesWidget: (value, meta) {
                          if (value.toInt() >= 0 &&
                              value.toInt() < sortedModels.length) {
                            String modelName = sortedModels[value.toInt()].key;
                            // Abbreviate long model names
                            if (modelName.length > 10) {
                              modelName = modelName.substring(0, 10) + '...';
                            }
                            return Padding(
                              padding: const EdgeInsets.only(top: 8.0),
                              child: Text(
                                modelName,
                                style: TextStyle(fontSize: 10),
                              ),
                            );
                          }
                          return Text('');
                        },
                        reservedSize: 30,
                      ),
                    ),
                  ),
                  borderData: FlBorderData(show: false),
                  barGroups: List.generate(
                    sortedModels.length,
                    (index) => BarChartGroupData(
                      x: index,
                      barRods: [
                        BarChartRodData(
                          toY: isClassification
                              ? sortedModels[index]
                                      .value
                                      .metrics[primaryMetric]! *
                                  100
                              : sortedModels[index]
                                  .value
                                  .metrics[primaryMetric]!,
                          color: _getBarColor(index, isClassification),
                          width: 20,
                          borderRadius: BorderRadius.vertical(
                            top: Radius.circular(4),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  double _getMaxValue(
      List<MapEntry<String, ModelResult>> models, String metric) {
    double maxVal = 0;
    for (var entry in models) {
      maxVal = maxVal > entry.value.metrics[metric]!
          ? maxVal
          : entry.value.metrics[metric]!;
    }
    return maxVal;
  }

  Color _getBarColor(int index, bool isClassification) {
    List<Color> colors = [
      Color(0xFF4CAF50), // Green
      Color(0xFF2196F3), // Blue
      Color(0xFFFFC107), // Amber
      Color(0xFFFF9800), // Orange
      Color(0xFFF44336), // Red
      Color(0xFF9C27B0), // Purple
      Color(0xFF795548), // Brown
      Color(0xFF607D8B), // Blue Grey
    ];

    if (isClassification) {
      // For classification, first bars (higher accuracy) are best
      return colors[index % colors.length];
    } else {
      // For regression, first bars (lower RMSE) are best
      return colors[index % colors.length];
    }
  }

  Widget _buildModelComparisonTable(BuildContext context,
      List<MapEntry<String, ModelResult>> sortedModels, bool isClassification) {
    // Get metrics that are common to all models
    Set<String> commonMetrics = {};
    if (sortedModels.isNotEmpty) {
      commonMetrics = sortedModels.first.value.metrics.keys.toSet();
      for (var entry in sortedModels.skip(1)) {
        commonMetrics =
            commonMetrics.intersection(entry.value.metrics.keys.toSet());
      }
    }

    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Detailed Model Comparison',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                columnSpacing: 20,
                dataRowHeight: 60,
                headingRowColor: MaterialStateProperty.all(
                  Colors.grey.shade200,
                ),
                columns: [
                  DataColumn(label: Text('Model')),
                  ...commonMetrics.map(
                    (metric) => DataColumn(
                      label: Text(
                        metric
                            .split('_')
                            .map((word) => word.capitalizeFirst())
                            .join(' '),
                        style: TextStyle(fontWeight: FontWeight.bold),
                      ),
                    ),
                  ),
                ],
                rows: sortedModels.map((entry) {
                  return DataRow(
                    cells: [
                      DataCell(
                        Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              entry.key,
                              style: TextStyle(fontWeight: FontWeight.bold),
                            ),
                            Text(
                              entry.value.explanation.substring(
                                      0,
                                      entry.value.explanation.length > 50
                                          ? 50
                                          : entry.value.explanation.length) +
                                  (entry.value.explanation.length > 50
                                      ? '...'
                                      : ''),
                              style: TextStyle(
                                fontSize: 12,
                                color: Colors.grey[600],
                              ),
                            ),
                          ],
                        ),
                      ),
                      ...commonMetrics.map((metric) {
                        double value = entry.value.metrics[metric]!;
                        // Format based on metric type
                        bool isPercentageMetric = metric == 'accuracy' ||
                            metric == 'precision' ||
                            metric == 'recall' ||
                            metric == 'f1_score' ||
                            metric == 'r2_score';
                        String formattedValue = isPercentageMetric
                            ? '${(value * 100).toStringAsFixed(2)}%'
                            : value.toStringAsFixed(4);

                        return DataCell(
                          Text(
                            formattedValue,
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: _getMetricColor(
                                  metric, value, isClassification),
                            ),
                          ),
                        );
                      }),
                    ],
                  );
                }).toList(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getMetricColor(String metric, double value, bool isClassification) {
    // For classification metrics (higher is better)
    if (metric == 'accuracy' ||
        metric == 'precision' ||
        metric == 'recall' ||
        metric == 'f1_score' ||
        metric == 'r2_score') {
      if (value > 0.8) return Colors.green;
      if (value > 0.6) return Colors.orange;
      return Colors.red;
    }

    // For regression error metrics (lower is better)
    if (metric == 'rmse' || metric.contains('error')) {
      if (value < 1.0) return Colors.green;
      if (value < 3.0) return Colors.orange;
      return Colors.red;
    }

    // Default
    return Colors.black;
  }
}
