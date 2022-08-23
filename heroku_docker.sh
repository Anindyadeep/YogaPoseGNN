heroku restart
heroku login 
heroku container:login
heroku container:push web -a yogaposegnn
heroku container:release web -a yogaposegnn
heroku open -a yogaposegnn