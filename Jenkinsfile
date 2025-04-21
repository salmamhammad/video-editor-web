pipeline {
    agent any

    environment {
        COMPOSE_FILE = 'docker-compose.yml'
    }

    stages {
        stage('Checkout Code') {
            steps {
                checkout scm
            }
        }

        stage('Build Docker Images') {
            steps {
                sh 'docker-compose build'
            }
        }

        stage('Start Services') {
            steps {
                sh 'docker-compose up -d'
                sh 'sleep 10'  // Wait for DB and services
            }
        }

        stage('Run Python Backend Tests') {
            steps {
                sh 'docker exec backend pytest backend/tests --disable-warnings'
            }
        }

        stage('Run Node.js  Tests') {
            steps {
                sh 'docker exec nodejs npm install'
                sh 'docker exec nodejs npm test'
            }
        }

        stage('Stop Services') {
            steps {
                sh 'docker-compose down'
            }
        }
    }

    post {
        always {
            echo 'Cleaning up...'
            sh 'docker-compose down -v'
        }
        success {
            echo '✅ All tests passed!'
        }
        failure {
            echo '❌ Some tests failed!'
        }
    }
}
