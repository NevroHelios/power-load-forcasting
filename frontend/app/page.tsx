"use client"

import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, PieChart, Pie, Cell } from 'recharts';
import { Activity, Cpu, Zap, TrendingUp, CheckCircle, AlertCircle, Clock, Target, Calendar } from 'lucide-react';

interface HealthStatus {
  status: string
  using: string | null
}

interface PredictionResult {
  prediction: string
  probabilities?: number[]
}

const Dashboard = () => {
  const BASE_URL = 'https://fastapi-ml-app-309739342275.europe-west1.run.app'; 
  // const BASE_URL = "http://localhost:8080";
  
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [currentModel, setCurrentModel] = useState('lgbm');
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState(false);
  // change history typing to accept the entries we push
  const [predictionHistory, setPredictionHistory] = useState<any[]>([]);
  const [error, setError] = useState<string | null>(null);

  // Form state for prediction
  const [formData, setFormData] = useState({
    Date_Time: '01-01-2018 00:15',
    Usage_kWh: 3.17,
    Lagging_Current_Reactive_Power_kVarh: 2.95,
    Leading_Current_Reactive_Power_kVarh: 0.0,
    CO2_tCO2: 0.0,
    Lagging_Current_Power_Factor: 73.21,
    Leading_Current_Power_Factor: 0.0,
    NSM: 25.0
  });

  // New: examples + expandable state to show public-facing examples for Medium/Maximum loads
  const [examplesExpanded, setExamplesExpanded] = useState(false);
  // New: engineered features expandable
  const [featuresExpanded, setFeaturesExpanded] = useState(false);

  const examples = [
    {
      id: 'ex1',
      label: 'Maximum_Load',
      Date_Time: '02-01-2019 19:15',
      Usage_kWh: 41.36,
      Lagging_Current_Reactive_Power_kVarh: 15.206443850474841,
      Leading_Current_Reactive_Power_kVarh: 0.43,
      CO2_tCO2: 0.0,
      Lagging_Current_Power_Factor: 98.81,
      Leading_Current_Power_Factor: 99.99,
      NSM: 69300.0
    },
    {
      id: 'ex2',
      label: 'Maximum_Load',
      Date_Time: '02-01-2019 19:30',
      Usage_kWh: 35.96,
      Lagging_Current_Reactive_Power_kVarh: 2.52,
      Leading_Current_Reactive_Power_kVarh: 1.37,
      CO2_tCO2: 0.0,
      Lagging_Current_Power_Factor: 99.76,
      Leading_Current_Power_Factor: 99.93,
      NSM: 138653.25999122107
    },
    {
      id: 'ex3',
      label: 'Medium_Load',
      Date_Time: '02-01-2019 15:30',
      Usage_kWh: 141.44,
      Lagging_Current_Reactive_Power_kVarh: 62.57,
      Leading_Current_Reactive_Power_kVarh: 0.0,
      CO2_tCO2: 0.0,
      Lagging_Current_Power_Factor: 91.45,
      Leading_Current_Power_Factor: 100.0,
      NSM: 55800.0
    },
    {
      id: 'ex4',
      label: 'Medium_Load',
      Date_Time: '02-01-2019 15:45',
      Usage_kWh: 116.53,
      Lagging_Current_Reactive_Power_kVarh: 52.88,
      Leading_Current_Reactive_Power_kVarh: 0.0,
      CO2_tCO2: 0.0,
      Lagging_Current_Power_Factor: 91.06,
      Leading_Current_Power_Factor: 100.0,
      NSM: 56700.0
    }
  ];

  // New: list of engineered / model features (display labels only)
  const engineeredFeatures: string[] = [
    'Usage_kWh',
    'Lagging_Current_Reactive.Power_kVarh',
    'Leading_Current_Reactive_Power_kVarh',
    'CO2(tCO2)',
    'Lagging_Current_Power_Factor',
    'Leading_Current_Power_Factor',
    'weekday',
    'dayofmonth',
    'month_sin',
    'month_cos',
    'hour',
    'hour_sin t/2',
    'hour_cos t/2',
    'hour_sin',
    'hour_cos',
    'hour_sin 2t',
    'hour_cos 2t',
    'hour_sin 3t',
    'hour_cos 3t',
    'hour_sin 4t',
    'hour_cos 4t',
    'dayofweek_sin',
    'dayofweek_cos',
    'total_reactive_power',
    'power_factor_diff',
    'reactive_usage_ratio',
    'isHoliday',
    'season',
    'usage_hour_sin',
    'usage_hour_cos'
  ];

  // Helper function to convert datetime-local to MM-DD-YYYY HH:MM format
  const formatDateTimeForAPI = (datetimeLocal: string) => {
    if (!datetimeLocal) return '';
    const date = new Date(datetimeLocal);
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    const year = date.getFullYear();
    const hours = String(date.getHours()).padStart(2, '0');
    const minutes = String(date.getMinutes()).padStart(2, '0');
    return `${month}-${day}-${year} ${hours}:${minutes}`;
  };

  // Helper function to convert MM-DD-YYYY HH:MM to datetime-local format
  const formatDateTimeForInput = (apiDateTime: string) => {
    if (!apiDateTime) return '';
    try {
      const [datePart, timePart] = apiDateTime.split(' ');
      const [month, day, year] = datePart.split('-');
      const [hours, minutes] = timePart.split(':');
      const date = new Date(parseInt(year), parseInt(month) - 1, parseInt(day), parseInt(hours), parseInt(minutes));
      return date.toISOString().slice(0, 16); // Format for datetime-local input
    } catch (error) {
      console.error('Error parsing date:', error);
      return '';
    }
  };

  // Real API functions
  const checkHealth = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BASE_URL}/health`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      const data = await response.json();
      setHealth(data);
      if (data.using) {
        setCurrentModel(data.using);
      }
    } catch (error) {
      console.error('Error checking health:', error);
      setError('Failed to check API health');
      setHealth({ status: 'error', using: null });
    }
    setLoading(false);
  };

  const changeModel = async (modelName: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${BASE_URL}/set-model`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ name: modelName })
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const data = await response.json();
      setCurrentModel(modelName);
      
      // Refresh health status after model change
      await checkHealth();
      
    } catch (error) {
      console.error('Error changing model:', error);
      setError(`Failed to change model to ${modelName}`);
    }
    setLoading(false);
  };

  const makePrediction = async (overrideData?: Partial<typeof formData>) => {
    setLoading(true);
    setError(null);
    try {
      // Use overrideData if provided (from examples) otherwise use current formData
      const src = overrideData ? overrideData : formData;
      const featuresData = {
        Date_Time: src.Date_Time,
        Usage_kWh: Number(src.Usage_kWh),
        Lagging_Current_Reactive_Power_kVarh: Number(src.Lagging_Current_Reactive_Power_kVarh),
        Leading_Current_Reactive_Power_kVarh: Number(src.Leading_Current_Reactive_Power_kVarh),
        CO2_tCO2: Number(src.CO2_tCO2),
        Lagging_Current_Power_Factor: Number(src.Lagging_Current_Power_Factor),
        Leading_Current_Power_Factor: Number(src.Leading_Current_Power_Factor),
        NSM: Number(src.NSM),
        get_features: [
          Number(src.Usage_kWh),
          Number(src.Lagging_Current_Reactive_Power_kVarh),
          Number(src.Leading_Current_Reactive_Power_kVarh),
          Number(src.CO2_tCO2),
          Number(src.Lagging_Current_Power_Factor),
          Number(src.Leading_Current_Power_Factor),
          Number(src.NSM)
        ]
      };
 
       const response = await fetch(`${BASE_URL}/predict`, {
         method: 'POST',
         headers: {
           'Content-Type': 'application/json',
         },
         body: JSON.stringify(featuresData)
       });
       
       if (!response.ok) {
         throw new Error(`HTTP error! status: ${response.status}`);
       }
       
       const data = await response.json();
       
       if (data.error) {
         throw new Error(data.error);
       }
       
       setPrediction(data);
       
       // Add to history
      const newEntry = {
        timestamp: new Date().toLocaleTimeString(),
        prediction: data.prediction,
        confidence: data.probabilities ? Math.max(...data.probabilities) * 100 : 0,
        usage: Number(src.Usage_kWh),
        sourceDateTime: src.Date_Time
      };
      setPredictionHistory(prev => [...prev.slice(-9), newEntry]);
       
     } catch (error: unknown) {
       let _error = error as Error;
       console.error('Error making prediction:', _error);
       setError(`Prediction failed: ${_error.message}`);
     }
     setLoading(false);
   };
  
  // Helper to apply an example into the form (and optionally run prediction)
  const applyExample = (ex: any, runPredict = false) => {
    setFormData({
      Date_Time: ex.Date_Time,
      Usage_kWh: ex.Usage_kWh,
      Lagging_Current_Reactive_Power_kVarh: ex.Lagging_Current_Reactive_Power_kVarh,
      Leading_Current_Reactive_Power_kVarh: ex.Leading_Current_Reactive_Power_kVarh,
      CO2_tCO2: ex.CO2_tCO2,
      Lagging_Current_Power_Factor: ex.Lagging_Current_Power_Factor,
      Leading_Current_Power_Factor: ex.Leading_Current_Power_Factor,
      NSM: ex.NSM
    });
    if (runPredict) {
      // call prediction with the example payload directly to avoid stale state issues
      makePrediction({
        Date_Time: ex.Date_Time,
        Usage_kWh: ex.Usage_kWh,
        Lagging_Current_Reactive_Power_kVarh: ex.Lagging_Current_Reactive_Power_kVarh,
        Leading_Current_Reactive_Power_kVarh: ex.Leading_Current_Reactive_Power_kVarh,
        CO2_tCO2: ex.CO2_tCO2,
        Lagging_Current_Power_Factor: ex.Lagging_Current_Power_Factor,
        Leading_Current_Power_Factor: ex.Leading_Current_Power_Factor,
        NSM: ex.NSM
      });
    }
  };

  useEffect(() => {
    checkHealth();
  }, []);

  // Performance metrics
  const metrics = [
    { label: 'F1 Score', value: '94%', icon: Target, color: 'text-green-500' },
    { label: 'Accuracy', value: '92%', icon: CheckCircle, color: 'text-blue-500' },
    { label: 'Latency', value: '572ms avg', icon: Clock, color: 'text-purple-500' },
    { label: 'Throughput', value: '25k/min', icon: TrendingUp, color: 'text-orange-500' }
  ];


  // Chart data
  const pieData = prediction?.probabilities ? [
    { name: 'Light Load', value: prediction.probabilities[0] * 100, color: '#10B981' },
    { name: 'Medium Load', value: prediction.probabilities[1] * 100, color: '#F59E0B' },
    { name: 'Maximum Load', value: prediction.probabilities[2] * 100, color: '#EF4444' }
  ] : [];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white p-3 sm:p-6">
      <div className="max-w-7xl mx-auto">
        
        {/* Header */}
        <div className="mb-6 sm:mb-8">
          <h1 className="text-2xl sm:text-4xl font-bold bg-gradient-to-r from-blue-400 to-purple-500 bg-clip-text text-transparent mb-2">
            ML Model Dashboard
          </h1>
          <p className="text-slate-400 text-sm sm:text-base">Real-time load prediction and model management</p>
          {error && (
            <div className="mt-3 bg-red-500/20 border border-red-500/50 rounded-lg p-3 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span className="text-red-400 text-xs sm:text-sm flex-1">{error}</span>
              <button 
                onClick={() => setError(null)}
                className="text-red-400 hover:text-red-300 text-lg leading-none"
              >
                ×
              </button>
            </div>
          )}
        </div>

        {/* Main Layout - Responsive Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 sm:gap-6">
          
          {/* Left Sidebar - Metrics (Full width on mobile, sidebar on desktop) */}
          <div className="lg:col-span-3 space-y-4 sm:space-y-6">
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
              <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                <Activity className="w-4 h-4 sm:w-5 sm:h-5 text-blue-400" />
                Performance Metrics
              </h3>
              <div className="grid grid-cols-2 lg:grid-cols-1 gap-3 sm:gap-4">
                {metrics.map((metric, index) => {
                  const Icon = metric.icon;
                  return (
                    <div key={index} className="flex items-center justify-between p-2 sm:p-3 bg-slate-700/30 rounded-lg">
                      <div className="flex items-center gap-2 sm:gap-3">
                        <Icon className={`w-3 h-3 sm:w-4 sm:h-4 ${metric.color}`} />
                        <span className="text-xs sm:text-sm text-slate-300">{metric.label}</span>
                      </div>
                      <span className={`font-semibold text-xs sm:text-sm ${metric.color}`}>{metric.value}</span>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Health Status */}
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
              <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                <Cpu className="w-4 h-4 sm:w-5 sm:h-5 text-green-400" />
                System Health
              </h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-slate-300 text-sm">Status</span>
                  <div className="flex items-center gap-2">
                    {health?.status === 'healthy' ? (
                      <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4 text-green-400" />
                    ) : (
                      <AlertCircle className="w-3 h-3 sm:w-4 sm:h-4 text-red-400" />
                    )}
                    <span className={`text-xs sm:text-sm ${health?.status === 'healthy' ? 'text-green-400' : 'text-red-400'}`}>
                      {health?.status || 'Checking...'}
                    </span>
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-slate-300 text-sm">Active Model</span>
                  <span className="text-blue-400 font-mono text-xs sm:text-sm">{health?.using || 'None'}</span>
                </div>
                <button
                  onClick={checkHealth}
                  disabled={loading}
                  className="w-full mt-3 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 transition-colors duration-200 py-2 px-3 sm:px-4 rounded-lg text-xs sm:text-sm font-medium"
                >
                  {loading ? 'Checking...' : 'Refresh Health'}
                </button>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="lg:col-span-9 space-y-4 sm:space-y-6">
            
            {/* Model Selection & Quick Prediction */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
              
              {/* Model Selection */}
              <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                  <Cpu className="w-4 h-4 sm:w-5 sm:h-5 text-purple-400" />
                  Model Selection
                </h3>
                <div className="space-y-3">
                  {['lgbm', 'xgboost'].map((model) => (
                    <button
                      key={model}
                      onClick={() => changeModel(model)}
                      disabled={loading || currentModel === model}
                      className={`w-full p-3 rounded-lg text-left transition-all duration-200 ${
                        currentModel === model
                          ? 'bg-purple-600 text-white'
                          : 'bg-slate-700/50 hover:bg-slate-700 text-slate-300'
                      } disabled:opacity-50`}
                    >
                      <div className="flex items-center justify-between">
                        <span className="font-medium capitalize text-sm sm:text-base">{model}</span>
                        {currentModel === model && <CheckCircle className="w-3 h-3 sm:w-4 sm:h-4" />}
                      </div>
                      <div className="text-xs text-slate-400 mt-1">
                        {model === 'lgbm' ? 'Light Gradient Boosting' : 'Extreme Gradient Boosting'}
                      </div>
                    </button>
                  ))}
                </div>
              </div>

              {/* Quick Prediction */}
              <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4 flex items-center gap-2">
                  <Zap className="w-4 h-4 sm:w-5 sm:h-5 text-yellow-400" />
                  Quick Predict
                </h3>
                <div className="space-y-3">
                  <div>
                    <label className="block text-xs sm:text-sm text-slate-300 mb-1">Usage (kWh)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.Usage_kWh}
                      onChange={(e) => setFormData({...formData, Usage_kWh: parseFloat(e.target.value)})}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label className="block text-xs sm:text-sm text-slate-300 mb-1">CO2 (tCO2)</label>
                    <input
                      type="number"
                      step="0.01"
                      value={formData.CO2_tCO2}
                      onChange={(e) => setFormData({...formData, CO2_tCO2: parseFloat(e.target.value)})}
                      className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                    />
                  </div>
                  <button
                    onClick={() => makePrediction()}
                    disabled={loading || health?.status !== 'healthy'}
                    className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:opacity-50 transition-all duration-200 py-2 sm:py-3 px-3 sm:px-4 rounded-lg font-medium text-sm sm:text-base"
                  >
                    {loading ? 'Predicting...' : 'Predict Load'}
                  </button>
                </div>
              </div>
            </div>

            {/* Prediction Results */}
            {prediction && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 sm:gap-6">
                <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
                  <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">Prediction Result</h3>
                  <div className="text-center">
                    <div className={`text-xl sm:text-3xl font-bold mb-2 ${
                      prediction.prediction === 'Light Load' ? 'text-green-400' :
                      prediction.prediction === 'Medium Load' ? 'text-yellow-400' : 'text-red-400'
                    }`}>
                      {prediction.prediction}
                    </div>
                    <div className="text-slate-400 text-sm">
                      Confidence: {prediction.probabilities ? `${(Math.max(...prediction.probabilities) * 100).toFixed(1)}%` : 'N/A'}
                    </div>
                  </div>
                </div>

                <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
                  <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">Probability Distribution</h3>
                  {prediction.probabilities && (
                    <ResponsiveContainer width="100%" height={160}>
                      <PieChart>
                        <Pie
                          data={pieData}
                          cx="50%"
                          cy="50%"
                          outerRadius={60}
                          dataKey="value"
                          label={({name, value}) => `${name}: ${typeof value === 'number' ? value.toFixed(1) : value}%`}
                          // labelStyle={{ fontSize: '10px', fill: '#F3F4F6' }}
                        >
                          {pieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip formatter={(value) => `${typeof value === 'number' ? value.toFixed(1) : value}%`} />
                      </PieChart>
                    </ResponsiveContainer>
                  )}
                </div>
              </div>
            )}

            {/* Detailed Input Form */}
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
              <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">Detailed Feature Input</h3>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4">
                
                {/* Date Time Picker */}
                <div className="sm:col-span-2 lg:col-span-1">
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1 flex items-center gap-1">
                    <Calendar className="w-3 h-3 sm:w-4 sm:h-4" />
                    Date & Time
                  </label>
                  <input
                    type="datetime-local"
                    value={formatDateTimeForInput(formData.Date_Time)}
                    onChange={(e) => {
                      const formattedDateTime = formatDateTimeForAPI(e.target.value);
                      setFormData({...formData, Date_Time: formattedDateTime});
                    }}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                  <div className="text-xs text-slate-500 mt-1">
                    Format: {formData.Date_Time}
                  </div>
                </div>

                <div>
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1">Usage (kWh)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.Usage_kWh}
                    onChange={(e) => setFormData({...formData, Usage_kWh: parseFloat(e.target.value)})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1">Lagging Reactive Power (kVarh)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.Lagging_Current_Reactive_Power_kVarh}
                    onChange={(e) => setFormData({...formData, Lagging_Current_Reactive_Power_kVarh: parseFloat(e.target.value)})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1">Leading Reactive Power (kVarh)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.Leading_Current_Reactive_Power_kVarh}
                    onChange={(e) => setFormData({...formData, Leading_Current_Reactive_Power_kVarh: parseFloat(e.target.value)})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1">CO2 (tCO2)</label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.CO2_tCO2}
                    onChange={(e) => setFormData({...formData, CO2_tCO2: parseFloat(e.target.value)})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1">Lagging Power Factor</label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.Lagging_Current_Power_Factor}
                    onChange={(e) => setFormData({...formData, Lagging_Current_Power_Factor: parseFloat(e.target.value)})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1">Leading Power Factor</label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.Leading_Current_Power_Factor}
                    onChange={(e) => setFormData({...formData, Leading_Current_Power_Factor: parseFloat(e.target.value)})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                </div>
                
                <div>
                  <label className="block text-xs sm:text-sm text-slate-300 mb-1">NSM</label>
                  <input
                    type="number"
                    step="0.01"
                    value={formData.NSM}
                    onChange={(e) => setFormData({...formData, NSM: parseFloat(e.target.value)})}
                    className="w-full bg-slate-700 border border-slate-600 rounded-lg px-3 py-2 text-white text-sm focus:border-blue-400 focus:outline-none"
                  />
                </div>
                
                <div className="sm:col-span-2 lg:col-span-3">
                  <button
                    onClick={() => makePrediction()}
                    disabled={loading || health?.status !== 'healthy'}
                    className="w-full bg-gradient-to-r from-green-600 to-blue-600 hover:from-green-700 hover:to-blue-700 disabled:opacity-50 transition-all duration-200 py-2 sm:py-3 px-3 sm:px-4 rounded-lg font-medium text-sm sm:text-base"
                  >
                    {loading ? 'Processing...' : 'Run Full Prediction'}
                  </button>
                </div>
              </div>
            </div>

            {/* New: Public-facing expandable examples explaining Medium / Maximum load */}
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-base sm:text-lg font-semibold flex items-center gap-2">
                  <TrendingUp className="w-4 h-4 text-orange-400" />
                  What produces Medium & Maximum Load
                </h3>
                <button
                  onClick={() => setExamplesExpanded(!examplesExpanded)}
                  className="text-xs bg-slate-700/40 px-2 py-1 rounded-md"
                >
                  {examplesExpanded ? 'Hide examples' : 'Show examples'}
                </button>
              </div>

              {examplesExpanded && (
                <div className="space-y-3">
                  <p className="text-slate-400 text-sm">
                    These real examples show feature combinations that produced Medium or Maximum load in the dataset.
                    Click "Apply" to load values into the form or "Apply & Predict" to run the model on the example.
                  </p>

                  {/* Group by label */}
                  {['Medium_Load', 'Maximum_Load'].map((label) => (
                    <div key={label} className="pt-2">
                      <div className="text-xs text-slate-300 font-semibold mb-2">{label.replace('_', ' ')}</div>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                        {examples.filter(e => e.label === label).map((ex) => (
                          <div key={ex.id} className="p-3 bg-slate-700/30 rounded-lg">
                            <div className="flex items-start justify-between">
                              <div>
                                <div className="text-sm font-medium">{ex.Date_Time}</div>
                                <div className="text-xs text-slate-400">Usage: <span className="font-semibold text-white">{ex.Usage_kWh}</span> kWh</div>
                                <div className="text-xs text-slate-400">Lagging Reactive: <span className="font-semibold text-white">{ex.Lagging_Current_Reactive_Power_kVarh}</span></div>
                                <div className="text-xs text-slate-400">Lagging PF: <span className="font-semibold text-white">{ex.Lagging_Current_Power_Factor}</span>%</div>
                                <div className="text-xs text-slate-400">NSM: <span className="font-semibold text-white">{ex.NSM}</span></div>
                              </div>
                              <div className="flex flex-col gap-2">
                                <button
                                  onClick={() => applyExample(ex, false)}
                                  className="text-xs bg-blue-600 hover:bg-blue-700 px-3 py-1 rounded-md"
                                >
                                  Apply
                                </button>
                                <button
                                  onClick={() => applyExample(ex, true)}
                                  className="text-xs bg-green-600 hover:bg-green-700 px-3 py-1 rounded-md"
                                >
                                  Apply & Predict
                                </button>
                              </div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* New: Engineered Feature Glossary */}
            <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
              <div className="flex items-center justify-between mb-3">
                <h3 className="text-base sm:text-lg font-semibold flex items-center gap-2">
                  <Activity className="w-4 h-4 text-blue-400" />
                  Feature Set Used by Model
                </h3>
                <button
                  onClick={() => setFeaturesExpanded(!featuresExpanded)}
                  className="text-xs bg-slate-700/40 px-2 py-1 rounded-md"
                >
                  {featuresExpanded ? 'Hide' : 'Show'}
                </button>
              </div>
              {featuresExpanded && (
                <div className="space-y-4">
                  <p className="text-slate-400 text-sm">
                    These raw + engineered features are fed into the model (cyclical encodings, ratios, seasonal & holiday flags).
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {engineeredFeatures.map(f => (
                      <span
                        key={f}
                        className="text-[10px] sm:text-xs bg-slate-700/50 border border-slate-600 px-2 py-1 rounded-md font-mono text-slate-200"
                        title={f.includes('sin') || f.includes('cos') ? 'Cyclical encoding' : f}
                      >
                        {f}
                      </span>
                    ))}
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-[11px] text-slate-400">
                    <div className="space-y-1">
                      <div><span className="text-slate-200">*_sin / *_cos:</span> cyclical transforms (periodic signals)</div>
                      <div><span className="text-slate-200">total_reactive_power:</span> derived reactive components sum</div>
                      <div><span className="text-slate-200">power_factor_diff:</span> lagging vs leading PF difference</div>
                      <div><span className="text-slate-200">reactive_usage_ratio:</span> reactive energy vs usage ratio</div>
                    </div>
                    <div className="space-y-1">
                      <div><span className="text-slate-200">isHoliday:</span> binary holiday flag</div>
                      <div><span className="text-slate-200">season:</span> encoded seasonal bucket</div>
                      <div><span className="text-slate-200">usage_hour_sin / cos:</span> usage-weighted hour signal</div>
                      <div><span className="text-slate-200">weekday / dayofmonth:</span> calendar context</div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Prediction History Chart */}
            {predictionHistory.length > 0 && (
              <div className="bg-slate-800/50 backdrop-blur border border-slate-700 rounded-xl p-4 sm:p-6">
                <h3 className="text-base sm:text-lg font-semibold mb-3 sm:mb-4">Prediction History</h3>
                <div className="h-64 sm:h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={predictionHistory}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="timestamp" 
                        stroke="#9CA3AF"
                        fontSize={10}
                        angle={-45}
                        textAnchor="end"
                        height={60}
                      />
                      <YAxis 
                        stroke="#9CA3AF"
                        fontSize={10}
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: '#1F2937',
                          border: '1px solid #374151',
                          borderRadius: '8px',
                          color: '#F3F4F6',
                          fontSize: '12px'
                        }}
                      />
                      <Line 
                        type="monotone" 
                        dataKey="confidence" 
                        stroke="#3B82F6" 
                        strokeWidth={2}
                        dot={{ fill: '#3B82F6', strokeWidth: 2, r: 3 }}
                        name="Confidence %"
                      />
                      <Line 
                        type="monotone" 
                        dataKey="usage" 
                        stroke="#10B981" 
                        strokeWidth={2}
                        dot={{ fill: '#10B981', strokeWidth: 2, r: 3 }}
                        name="Usage kWh"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            )}

          </div>
        </div>

        {/* Footer Status Bar */}
        <div className="mt-8 bg-slate-800/30 backdrop-blur border border-slate-700 rounded-xl p-4">
          <div className="flex items-center justify-between text-sm">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-2">
                <div className={`w-2 h-2 rounded-full ${health?.status === 'healthy' ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-slate-300">API Status: {health?.status || 'Unknown'}</span>
              </div>
              <div className="text-slate-400">|</div>
              <span className="text-slate-300">Model: {currentModel.toUpperCase()}</span>
            </div>
            <div className="text-slate-400">
              Last updated: {new Date().toLocaleTimeString()}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
