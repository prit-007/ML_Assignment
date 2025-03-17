import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/foundation.dart' show kDebugMode, kIsWeb;
import 'package:flutter/material.dart';

import '../models/analysis_result.dart';
import '../services/api_service.dart';
import 'result_screen.dart';

class HomeScreen extends StatefulWidget {
  @override
  _HomeScreenState createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  bool _isLoading = false;
  File? _selectedFile;
  final ApiService _apiService = ApiService();
  Uint8List? _selectedFileBytes;
  String? _selectedFileName;
  @override
  void initState() {
    super.initState();
    _checkBackendConnection();
  }

  Future<void> _checkBackendConnection() async {
    try {
      final isHealthy = await _apiService.checkHealth();
      print("Backend health check: ${isHealthy ? 'SUCCESS' : 'FAILED'}");

      if (!isHealthy && mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Warning: Cannot connect to analysis server')),
        );
      }
    } catch (e) {
      print("Health check error: $e");
    }
  }

  Future<void> _pickFile() async {
    FilePickerResult? result = await FilePicker.platform.pickFiles(
      type: FileType.custom,
      allowedExtensions: ['csv'],
    );

    if (result != null) {
      // For web platform
      if (kIsWeb) {
        // Store bytes instead of File
        final bytes = result.files.single.bytes;
        // You might want to store filename too
        final fileName = result.files.single.name;

        setState(() {
          // Store the bytes in a state variable
          _selectedFileBytes = bytes;
          _selectedFileName = fileName;
          // Clear the File object since we're not using it
          _selectedFile = null;
        });
      } else {
        // For mobile/desktop platforms
        setState(() {
          _selectedFile = File(result.files.single.path!);
        });
      }
    }
  }

  Future<void> _uploadFile() async {
    if (_selectedFile == null && _selectedFileBytes == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Please select a CSV file first')),
      );
      return;
    }

    setState(() {
      _isLoading = true;
    });

    try {
      print("Starting file upload...");
      var healthCheck = await _apiService.checkHealth();
      print("Health check result: $healthCheck");

      AnalysisResult result;

      if (kIsWeb) {
        print(
            "Using web upload method with bytes length: ${_selectedFileBytes?.length}");
        if (_selectedFileBytes != null && _selectedFileName != null) {
          // Validate file extension
          if (!_selectedFileName!.toLowerCase().endsWith('.csv')) {
            throw "Please upload a valid CSV file";
          }
          result = await _apiService.uploadFileBytes(
              _selectedFileBytes!, _selectedFileName!);
        } else {
          throw "File bytes or name is null";
        }
      } else {
        print(
            "Using native upload method with file path: ${_selectedFile?.path}");
        if (_selectedFile != null) {
          // Validate file extension
          if (!_selectedFile!.path.toLowerCase().endsWith('.csv')) {
            throw "Please upload a valid CSV file";
          }
          result = await _apiService.uploadFile(_selectedFile!);
        } else {
          throw "File is null";
        }
      }

      print("API call completed");
      print("Result: $result");

      if (!mounted) return;

      Navigator.push(
        context,
        MaterialPageRoute(
          builder: (context) => ResultsScreen(result: result),
        ),
      );
    } catch (e) {
      print("Error during upload: ${e.toString()}");
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Error: ${e.toString()}')),
      );
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('ML Model Analyzer'),
        elevation: 0,
      ),
      body: Center(
        child: _isLoading
            ? LoadingWidget()
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(
                    Icons.analytics,
                    size: 80,
                    color: Theme.of(context).primaryColor,
                  ),
                  SizedBox(height: 24),
                  Text(
                    'Upload a CSV file to analyze',
                    style: TextStyle(fontSize: 18),
                  ),
                  SizedBox(height: 32),
                  FileSelectorWidget(
                    fileName: _selectedFile?.path.split('/').last ??
                        _selectedFileName,
                    onPressed: _pickFile,
                  ),
                  SizedBox(height: 24),
                  if (_selectedFile != null || _selectedFileBytes != null)
                    ElevatedButton.icon(
                      icon: Icon(Icons.upload),
                      label: Text('Analyze Data'),
                      style: ElevatedButton.styleFrom(
                        padding: EdgeInsets.symmetric(
                          horizontal: 32,
                          vertical: 16,
                        ),
                      ),
                      onPressed: _uploadFile,
                    ),
                  SizedBox(height: 16),
                  Text(
                    'Your data will be processed securely',
                    style: TextStyle(
                      color: Colors.grey,
                      fontSize: 12,
                    ),
                  ),
                ],
              ),
      ),
    );
  }
}

class FileSelectorWidget extends StatelessWidget {
  final String? fileName;
  final VoidCallback onPressed;

  const FileSelectorWidget({
    Key? key,
    this.fileName,
    required this.onPressed,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        OutlinedButton.icon(
          icon: Icon(Icons.file_upload),
          label: Text('Select CSV File'),
          style: OutlinedButton.styleFrom(
            padding: EdgeInsets.symmetric(horizontal: 24, vertical: 16),
          ),
          onPressed: onPressed,
        ),
        if (fileName != null)
          Padding(
            padding: const EdgeInsets.only(top: 8.0),
            child: Text(
              'Selected: $fileName',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
          ),
      ],
    );
  }
}

class LoadingWidget extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        CircularProgressIndicator(),
        SizedBox(height: 24),
        Text(
          'Analyzing your data...\nThis may take a moment',
          textAlign: TextAlign.center,
          style: TextStyle(fontSize: 16),
        ),
      ],
    );
  }
}
