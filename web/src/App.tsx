import { Route, Routes } from 'react-router-dom'
import Shell from './components/Shell'
import Overview from './pages/Overview'
import Groups from './pages/Groups'
import CustomerPage from './pages/Customer'
import CampaignPage from './pages/Campaign'
import Results from './pages/Results'
import HowItWorks from './pages/HowItWorks'
import Guide from './pages/Guide'

export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Overview />} />
        <Route path="/grupos" element={<Groups />} />
        <Route path="/cliente" element={<CustomerPage />} />
        <Route path="/campana" element={<CampaignPage />} />
        <Route path="/resultados" element={<Results />} />
        <Route path="/como-funciona" element={<HowItWorks />} />
        <Route path="/guia" element={<Guide />} />
        <Route path="*" element={<Overview />} />
      </Routes>
    </Shell>
  )
}
