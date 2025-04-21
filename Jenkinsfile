pipeline {
    agent any

    environment {
        COMPOSE_FILE = 'docker-compose.yml'
    }

    stages {
        stage('📦 Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('🐳 Build Docker Images') {
            steps {
                bat 'docker-compose build'
            }
        }

        stage('🚀 Start Services') {
            steps {
                bat 'docker-compose up -d'
                bat 'timeout /t 15'  // Windows equivalent of sleep
            }
        }

        stage('🧪 Run Python Backend Tests') {
            steps {
                bat 'docker exec backend pip install -r backend/requirements.txt'
                bat 'docker exec backend pytest backend/tests --disable-warnings --maxfail=1'
            }
        }

        stage('🧪 Run Node.js Tests') {
            steps {
                bat 'docker exec nodejs npm install'
                bat 'docker exec nodejs npm test'
            }
        }

        stage('🛑 Stop Services') {
            steps {
                bat 'docker-compose down'
            }
        }
    }

    post {
        always {
            echo '🧹 Cleaning up containers and volumes...'
            bat 'docker-compose down -v'
        }
        success {
            echo '✅ All tests passed! Nice job!'
        }
        failure {
            echo '❌ Some tests failed! Check logs above.'
        }
    }
}
