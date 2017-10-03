#!/bin/bash

#Author: Hongyuan Mei

# Process the command line options
while true; do
  [ -z "${1:-}" ] && break;  # break if there are no arguments
  case "$1" in
  # deal with the argument begins with "--". Only support the format: --name value.
  --*=*) echo "$0: options to scripts must be of the form --name value, rather than --name=value, got '$1'"
       exit 1 ;;
  #then work out the variable name as $name, and convert the "-" to "_".
  --*) name=`echo "$1" | sed s/^--// | sed s/-/_/g`;
      # Next we test whether the variable in question is undefined, if so it's
      # an invalid option and we die.

      # The test [ -z ${var+xxx} ] will return true if the variable var
      # is undefined.  We then have to wrap this test inside "eval" because
      # var is itself inside a variable ($name).
      eval '[ -z "${'$name'+xxx}" ]' && echo "$0: invalid option $1" 1>&2 && exit 1;

      oldval="`eval echo \\$$name`";
      if [ "$oldval" == "true" ] || [ "$oldval" == "false" ]; then
	    was_bool=true;
      else
	    was_bool=false;
      fi
      # Set the variable to the right value, the escaped quotes make it work if
      # the option had spaces.
      eval $name=\"$2\";

      # Check that Boolean-valued arguments.
      if $was_bool && [[ "$2" != "true" && "$2" != "false" ]]; then
        echo "$0: expected \"true\" or \"false\": $1 $2" 1>&2
        exit 1;
      fi
      shift 2;
      ;;
  *) break;
  esac
done

true; 
