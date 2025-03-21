import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../models/analysis_result.dart';

class FeatureImportanceWidget extends StatelessWidget {
  final String bestModel;
  final Map<String, ModelResult> allResults;

  FeatureImportanceWidget({required this.bestModel, required this.allResults});

  @override
  Widget build(BuildContext context) {
    if (!allResults.containsKey(bestModel) ||
        allResults[bestModel]!.featureImportance == null ||
        allResults[bestModel]!.featureImportance!.isEmpty) {
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Icon(Icons.info_outline, size: 48, color: Colors.grey),
            SizedBox(height: 16),
            Text(
              'Feature importance not available for this model',
              style: TextStyle(color: Colors.grey[600]),
            ),
          ],
        ),
      );
    }

    var featureImportance = allResults[bestModel]!.featureImportance!;
    var sortedFeatures = featureImportance.entries.toList()
      ..sort((a, b) => b.value.compareTo(a.value));

    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildFeatureImportanceChart(context, sortedFeatures),
          SizedBox(height: 24),
          _buildFeatureImportanceList(context, sortedFeatures),
          SizedBox(height: 24),
          _buildFeatureGuideCard(context),
        ],
      ),
    );
  }

  Widget _buildFeatureImportanceChart(
      BuildContext context, List<MapEntry<String, double>> sortedFeatures) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Feature Importance',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 8),
            Text(
              'Top features for $bestModel',
              style: TextStyle(color: Colors.grey[600]),
            ),
            SizedBox(height: 16),
            Container(
              height: 300,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: sortedFeatures.first.value * 1.2,
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
                              value.toInt() < sortedFeatures.length) {
                            String featureName =
                                sortedFeatures[value.toInt()].key;
                            // Abbreviate long feature names
                            if (featureName.length > 10) {
                              featureName =
                                  featureName.substring(0, 10) + '...';
                            }
                            return Padding(
                              padding: const EdgeInsets.only(top: 8.0),
                              child: Text(
                                featureName,
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
                    sortedFeatures.length,
                    (index) => BarChartGroupData(
                      x: index,
                      barRods: [
                        BarChartRodData(
                          toY: sortedFeatures[index].value,
                          color:
                              _getGradientColor(index, sortedFeatures.length),
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

  Color _getGradientColor(int index, int total) {
    // Color gradient from dark to light blue
    final colors = [
      Color(0xFF1565C0), // Dark blue
      Color(0xFF1976D2),
      Color(0xFF1E88E5),
      Color(0xFF2196F3),
      Color(0xFF42A5F5),
      Color(0xFF64B5F6),
      Color(0xFF90CAF9), // Light blue
    ];

    int colorIndex =
        (index * colors.length ~/ total).clamp(0, colors.length - 1);
    return colors[colorIndex];
  }

  Widget _buildFeatureImportanceList(
      BuildContext context, List<MapEntry<String, double>> sortedFeatures) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Feature Ranking',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            ListView.builder(
              shrinkWrap: true,
              physics: NeverScrollableScrollPhysics(),
              itemCount: sortedFeatures.length,
              itemBuilder: (context, index) {
                final feature = sortedFeatures[index];
                final percentage = (feature.value * 100).toStringAsFixed(1);
                return ListTile(
                  contentPadding: EdgeInsets.zero,
                  leading: CircleAvatar(
                    backgroundColor:
                        _getGradientColor(index, sortedFeatures.length),
                    child: Text(
                      '${index + 1}',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                  title: Text(feature.key),
                  trailing: Text(
                    '$percentage%',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ),
                  subtitle: LinearProgressIndicator(
                    value: feature.value / sortedFeatures.first.value,
                    backgroundColor: Colors.grey.shade200,
                    valueColor: AlwaysStoppedAnimation<Color>(
                      _getGradientColor(index, sortedFeatures.length),
                    ),
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureGuideCard(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'How to Use Feature Importance',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            ListTile(
              leading: Icon(Icons.lightbulb, color: Colors.amber),
              title: Text('Focus on High-Impact Features'),
              subtitle: Text(
                  'Features with high importance have the most influence on predictions.'),
            ),
            ListTile(
              leading: Icon(Icons.filter_list, color: Colors.blue),
              title: Text('Feature Selection'),
              subtitle: Text(
                  'Consider removing low importance features to simplify your model.'),
            ),
            ListTile(
              leading: Icon(Icons.insights, color: Colors.purple),
              title: Text('Feature Engineering'),
              subtitle: Text(
                  'Create new features based on the most important ones to improve performance.'),
            ),
          ],
        ),
      ),
    );
  }
}
