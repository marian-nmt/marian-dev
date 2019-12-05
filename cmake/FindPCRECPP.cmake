# - Try to find PCRE
# Once done this will define
#
#  PCRE_FOUND        - system has PCRE
#  PCRE_INCLUDE_DIR  - the PCRE include directory
#  PCRE_LIBRARY      - Link these to use PCRE
#

IF (PCRE_INCLUDE_DIRS)
  # Already in cache, be silent
  SET(PCRE_FIND_QUIETLY TRUE)
ENDIF (PCRE_INCLUDE_DIRS)

FIND_PATH( PCRE_INCLUDE_DIR pcre.h
           PATHS "/usr/include" "C:/libs/PCRE/include")

if( WIN32 )

 FIND_LIBRARY( PCRE_LIBRARY
               NAMES pcrecpp.lib
               PATHS "C:/libs/PCRE/lib")

 # Store the library dir. May be used for linking to dll!
 GET_FILENAME_COMPONENT( PCRE_LIBRARY_DIR ${PCRE_LIBRARY} PATH )
else (WIN32)

 FIND_LIBRARY( PCRE_LIBRARY
               NAMES pcrecpp.a pcrecpp
               PATHS /lib /usr/lib /usr/local/lib )

endif( WIN32)


IF (PCRE_INCLUDE_DIR AND PCRE_LIBRARY)
  SET(PCRE_FOUND TRUE)
ELSE (PCRE_INCLUDE_DIR AND PCRE_LIBRARY)
  SET( PCRE_FOUND FALSE )
ENDIF (PCRE_INCLUDE_DIR AND PCRE_LIBRARY)
