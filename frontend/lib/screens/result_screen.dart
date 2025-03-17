import 'package:flutter/material.dart';

import '../models/analysis_result.dart';
import '../widgets/data_insights_widget.dart';
import '../widgets/feature_importance_widget.dart';
import '../widgets/model_comparison_widget.dart';
import '../widgets/model_recommendation_widget.dart';

class ResultsScreen extends StatefulWidget {
  final AnalysisResult result;

  const ResultsScreen({super.key, required this.result});

  @override
  _ResultsScreenState createState() => _ResultsScreenState();
}

class _ResultsScreenState extends State<ResultsScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 4, vsync: this);
  }

  @override
  void dispose() {
    _tabController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Analysis Results'),
        bottom: TabBar(
          controller: _tabController,
          tabs: [
            Tab(text: 'Data Insights', icon: Icon(Icons.analytics)),
            Tab(text: 'Best Model', icon: Icon(Icons.recommend)),
            Tab(text: 'Comparison', icon: Icon(Icons.compare)),
            Tab(text: 'Features', icon: Icon(Icons.insights)),
          ],
        ),
      ),
      body: TabBarView(
        controller: _tabController,
        children: [
          DataInsightsWidget(insights: widget.result.insights),
          ModelRecommendationWidget(
              recommendation: widget.result.modelRecommendation),
          ModelComparisonWidget(
              modelResults: widget.result.modelRecommendation.allResults),
          FeatureImportanceWidget(
            bestModel: widget.result.modelRecommendation.bestModel,
            allResults: widget.result.modelRecommendation.allResults,
          ),
        ],
      ),
    );
  }
}
