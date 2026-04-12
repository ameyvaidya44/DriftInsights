import MetricsCards from '../components/MetricsCards'
import AccuracyChart from '../components/AccuracyChart'
import FeatureImportance from '../components/FeatureImportance'
import DriftTimeline from '../components/DriftTimeline'
import ModelInfo from '../components/ModelInfo'
import PipelineControls from '../components/PipelineControls'
import PipelineFlowViz from '../components/dashboard/PipelineFlowViz'
import LiveDecisionFeed from '../components/dashboard/LiveDecisionFeed'
import SystemHealthRow from '../components/dashboard/SystemHealthRow'
import useFetch from '../hooks/useFetch'
import { fetchAllDashboardData } from '../services/api'

const Skeleton = () => (
  <div className="animate-pulse space-y-3">
    <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-3/4"></div>
    <div className="h-32 bg-gray-200 dark:bg-gray-700 rounded"></div>
  </div>
)

export default function Dashboard() {
  const { data, loading, error, refetch } = useFetch(fetchAllDashboardData, { refreshInterval: 3000 })

  if (loading && !data) return (
    <div className="page-dashboard">
      <div className="page-header">
        <h2 className="page-title">Dashboard</h2>
      </div>
      <Skeleton />
    </div>
  )

  if (error && !data) return (
    <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg text-red-700 dark:text-red-400 text-sm m-4">
      <span>Failed to load dashboard data</span>
      <button onClick={refetch} className="ml-auto underline">Retry</button>
    </div>
  )

  const { status, metrics, metricsHistory, driftEvents, modelInfo, modelHistory } = data || {}

  return (
    <div className="page-dashboard">
      <div className="page-header">
        <h2 className="page-title">Dashboard</h2>
        <p className="page-desc">Real-time overview of the DriftInsights ML pipeline</p>
      </div>

      <MetricsCards metrics={metrics} status={status} />
      <PipelineFlowViz status={status} />
      <SystemHealthRow status={status} modelInfo={modelInfo} />

      <div className="grid-2col-wide">
        <AccuracyChart history={metricsHistory} driftEvents={driftEvents} />
        <FeatureImportance />
      </div>

      <div className="grid-2col">
        <DriftTimeline events={driftEvents} />
        <div>
          <ModelInfo info={modelInfo} history={modelHistory} />
          <LiveDecisionFeed />
        </div>
      </div>

      <PipelineControls onRefresh={refetch} status={status} />
    </div>
  )
}
