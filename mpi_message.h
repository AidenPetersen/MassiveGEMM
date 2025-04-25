#pragma once

#include <mpi.h>

#define CLIENT_REQUEST_WL 0
#define CLIENT_RETURN_WL 1

/**
 * General handshake sequences
 * CLIENT_REQUEST_WL -> SERVER_REPLY_WL -> CLIENT_RETURN_WL -> client sends big array of data over
 * CLIENT_RETURN_WL acts as a prep signal for the next message to accept a lot of data
 */
