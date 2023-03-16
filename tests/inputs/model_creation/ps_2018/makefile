s := false

git:
	git pull
	git add .
	git commit -m "$m"
	git push origin HEAD -o ci.variable="CONDOR_SUBMIT=$s"