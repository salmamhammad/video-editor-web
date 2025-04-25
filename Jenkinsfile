pipeline {
    agent any

    environment {
        COMPOSE_FILE = 'docker-compose.yml'
    }
    stages {
            stage('Install Docker CLI') {
            steps {
                sh '''
                    if ! command -v docker > /dev/null 2>&1; then
                      echo "Installing Docker CLI..."
                      apt-get update
                      apt-get install -y docker.io
                    else
                      echo "Docker already installed"
                    fi
                '''
            }
        }

        stage('Check Docker Version') {
            steps {
                sh 'docker --version'
            }
        }

        stage('Install Docker Compose') {
            steps {
                sh '''
                    if ! command -v docker-compose >/dev/null 2>&1; then
                      echo "Installing Docker Compose..."
                      curl -L "https://github.com/docker/compose/releases/download/v2.24.5/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
                      chmod +x /usr/local/bin/docker-compose
                      ln -s /usr/local/bin/docker-compose /usr/bin/docker-compose || true
                    else
                      echo "Docker Compose already installed"
                    fi
                '''
            }
        }
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
        stage('🚀 Start Service Backend') {
            steps {
                sh 'docker-compose up -d backend'
                sh 'sleep 60'
            }
        }
        stage('🚀 Start Service  Node.js') {
            steps {
                sh  'docker-compose up -d nodejs'
                sh  'sleep 5'  // Windows equivalent of sleep
            }
        }
         stage('🧪 Run Python Backend Tests') {
            steps {
                sh  'docker exec backendweb pytest tests --disable-warnings --maxfail=1'
            }
        }
        stage('🧪 Run Node.js Tests') {
            steps {
                sh 'docker exec nodejsweb npm install'
                sh 'docker exec nodejsweb npm install --save-dev wait-on supertest'
                sh  'docker exec nodejsweb npx jest full-server.test.js'
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
