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
                sh  'docker-compose build'
            }
        }

        stage('🚀 Start Services') {
            steps {
                sh  'docker-compose up -d'
                sh  'timeout /t 15'  // Windows equivalent of sleep
            }
        }

        stage('🧪 Run Python Backend Tests') {
            steps {
                sh  'docker exec backend pip install -r backend/requirements.txt'
                sh  'docker exec backend pytest backend/tests --disable-warnings --maxfail=1'
            }
        }

        stage('🧪 Run Node.js Tests') {
            steps {
                sh  'docker exec nodejs npm install'
                sh  'docker exec nodejs npm test'
            }
        }

        stage('🛑 Stop Services') {
            steps {
                sh  'docker-compose down'
            }
        }
    }

    post {
        always {
            echo '🧹 Cleaning up containers and volumes...'
            sh  'docker-compose down -v'
        }
        success {
            echo '✅ All tests passed! Nice job!'
        }
        failure {
            echo '❌ Some tests failed! Check logs above.'
        }
    }
}
