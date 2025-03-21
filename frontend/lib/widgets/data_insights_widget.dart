import 'dart:math';

import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

import '../models/analysis_result.dart';

class DataInsightsWidget extends StatelessWidget {
  final DataInsights insights;

  DataInsightsWidget({required this.insights});

  @override
  Widget build(BuildContext context) {
    return SingleChildScrollView(
      padding: EdgeInsets.all(16),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildDatasetOverview(context),
          SizedBox(height: 24),
          _buildMissingValuesCard(context),
          SizedBox(height: 24),
          if (insights.classDistribution != null)
            _buildClassDistribution(context),
          SizedBox(height: 24),
          _buildHighCorrelations(context),
          SizedBox(height: 24),
          _buildNumericalStats(context),
          SizedBox(height: 24),
          _buildPreprocessingSuggestions(context),
        ],
      ),
    );
  }

  Widget _buildDatasetOverview(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Dataset Overview',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            _buildOverviewRow('Rows', insights.rows.toString()),
            _buildOverviewRow('Columns', insights.columns.toString()),
            _buildOverviewRow('Categorical Features',
                insights.categoricalFeatures.toString()),
            _buildOverviewRow(
                'Numerical Features', insights.numericalFeatures.toString()),
            _buildOverviewRow('Target Column', insights.targetColumn),
          ],
        ),
      ),
    );
  }

  Widget _buildOverviewRow(String label, String value) {
    return Padding(
      padding: EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label),
          Text(value, style: TextStyle(fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildMissingValuesCard(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Missing Values',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  flex: 2,
                  child: _buildCircularPercentIndicator(
                    insights.missingPercentage.toDouble(),
                    '${insights.missingPercentage.toStringAsFixed(1)}%',
                    Color(0xFF2196F3),
                  ),
                ),
                Expanded(
                  flex: 3,
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Total Missing Values: ${insights.missingValues}'),
                      SizedBox(height: 8),
                      Text(
                        insights.missingPercentage > 5
                            ? 'Significant missing data detected'
                            : 'Low level of missing data',
                        style: TextStyle(
                          color: insights.missingPercentage > 5
                              ? Colors.orange
                              : Colors.green,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildCircularPercentIndicator(
      double percent, String centerText, Color color) {
    return SizedBox(
      height: 100,
      width: 100,
      child: Stack(
        alignment: Alignment.center,
        children: [
          PieChart(
            PieChartData(
              sectionsSpace: 0,
              centerSpaceRadius: 35,
              sections: [
                PieChartSectionData(
                  color: color,
                  value: percent,
                  title: '',
                  radius: 10,
                ),
                PieChartSectionData(
                  color: Colors.grey.shade300,
                  value: 100 - percent,
                  title: '',
                  radius: 10,
                ),
              ],
            ),
          ),
          Text(
            centerText,
            style: TextStyle(
              fontSize: 16,
              fontWeight: FontWeight.bold,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildClassDistribution(BuildContext context) {
    Map<String, dynamic> distribution = insights.classDistribution!;
    List<MapEntry<String, dynamic>> entries = distribution.entries.toList();

    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Class Distribution',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            AspectRatio(
              aspectRatio: 1.5,
              child: BarChart(
                BarChartData(
                  alignment: BarChartAlignment.spaceAround,
                  maxY: entries
                          .map((e) => e.value as num)
                          .reduce((a, b) => a > b ? a : b)
                          .toDouble() *
                      1.2,
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
                              value.toInt() < entries.length) {
                            return Padding(
                              padding: const EdgeInsets.only(top: 8.0),
                              child: Text(
                                entries[value.toInt()].key.toString().substring(
                                    0,
                                    min(
                                        6,
                                        entries[value.toInt()]
                                            .key
                                            .toString()
                                            .length)),
                                style: TextStyle(fontSize: 12),
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
                    entries.length,
                    (index) => BarChartGroupData(
                      x: index,
                      barRods: [
                        BarChartRodData(
                          toY: entries[index].value.toDouble(),
                          color: Color(0xFF2196F3),
                          width: 20,
                          borderRadius:
                              BorderRadius.vertical(top: Radius.circular(4)),
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

  Widget _buildHighCorrelations(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'High Feature Correlations',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            insights.highCorrelations.isEmpty
                ? Text('No high correlations found')
                : ListView.builder(
                    shrinkWrap: true,
                    physics: NeverScrollableScrollPhysics(),
                    itemCount: insights.highCorrelations.length,
                    itemBuilder: (context, index) {
                      HighCorrelation corr = insights.highCorrelations[index];
                      return ListTile(
                        contentPadding: EdgeInsets.zero,
                        title: Text('${corr.feature1} & ${corr.feature2}'),
                        subtitle: LinearProgressIndicator(
                          value: corr.correlation,
                          backgroundColor: Colors.grey.shade200,
                          valueColor: AlwaysStoppedAnimation<Color>(
                              corr.correlation > 0.8
                                  ? Colors.red
                                  : Colors.orange),
                        ),
                        trailing: Text(
                          '${(corr.correlation * 100).toStringAsFixed(0)}%',
                          style: TextStyle(fontWeight: FontWeight.bold),
                        ),
                      );
                    },
                  ),
          ],
        ),
      ),
    );
  }

  Widget _buildNumericalStats(BuildContext context) {
    if (insights.numericalStats.isEmpty) {
      return SizedBox.shrink();
    }

    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Numerical Features Statistics',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: DataTable(
                columnSpacing: 20,
                columns: [
                  DataColumn(label: Text('Feature')),
                  DataColumn(label: Text('Min')),
                  DataColumn(label: Text('Max')),
                  DataColumn(label: Text('Mean')),
                  DataColumn(label: Text('Median')),
                  DataColumn(label: Text('Std Dev')),
                ],
                rows: insights.numericalStats.entries.map((entry) {
                  NumericalStat stat = entry.value;
                  return DataRow(cells: [
                    DataCell(Text(entry.key)),
                    DataCell(Text(stat.min.toStringAsFixed(2))),
                    DataCell(Text(stat.max.toStringAsFixed(2))),
                    DataCell(Text(stat.mean.toStringAsFixed(2))),
                    DataCell(Text(stat.median.toStringAsFixed(2))),
                    DataCell(Text(stat.std.toStringAsFixed(2))),
                  ]);
                }).toList(),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildPreprocessingSuggestions(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Preprocessing Suggestions',
              style: Theme.of(context).textTheme.titleLarge,
            ),
            SizedBox(height: 16),
            insights.preprocessingSuggestions.isEmpty
                ? Text('No preprocessing suggestions')
                : ListView.builder(
                    shrinkWrap: true,
                    physics: NeverScrollableScrollPhysics(),
                    itemCount: insights.preprocessingSuggestions.length,
                    itemBuilder: (context, index) {
                      return ListTile(
                        contentPadding: EdgeInsets.zero,
                        leading: Icon(Icons.lightbulb, color: Colors.amber),
                        title: Text(insights.preprocessingSuggestions[index]),
                      );
                    },
                  ),
          ],
        ),
      ),
    );
  }
}
