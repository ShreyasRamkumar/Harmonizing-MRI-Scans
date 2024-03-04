Jenkinsfile (Declarative Pipeline)
pipeline {
    agent {
        docker { image 'shreyasramkumar/harmonizing-mri-scans'}
    }
    stages {
        stage('Build') {
            steps {
                echo 'Building...'
            }
        }
        stage('Test') {
            steps {
                echo 'Testing...'                
            }
        }
        stage('Deploy') {
            steps {
                echo 'Deploying....'
            }
        }
    }
}