import React from 'react'
import ReactDOM from 'react-dom'
import './index.css'
import App from './App'
import { AuthProvider } from './context/AuthContext'
import { GamificationProvider } from './context/GamificationContext'

ReactDOM.render(
    <React.StrictMode>
        <AuthProvider>
            <GamificationProvider>
                <App />
            </GamificationProvider>
        </AuthProvider>
    </React.StrictMode>,
    document.getElementById('root')
)
