localinstall:
	rm -Rf build
	${CMP1_PYLOCALEXEC} setup.py install --home=${CMP1_PYLOCALDIR}
	rm -Rf build
