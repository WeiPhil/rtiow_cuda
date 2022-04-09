#pragma once

#ifndef BUFFERMANAGER_H
#define BUFFERMANAGER_H

#include <GL/glew.h>

#include <iostream>
#include <map>
#include <vector>
#include <assert.h>

class BufferManager
{

private:
  int m_max;
  std::map<std::string, int> m_buffers_map;

public:
  BufferManager() : m_max(0){};
  ~BufferManager(){};

  std::vector<GLuint> buffers;

  int size() const { return m_max; }

  void addBuffer(std::string bufferName)
  {
    m_buffers_map[bufferName] = m_max;
    ++m_max;
    buffers.resize(m_max);
    // for (auto const &a : buffers) {
    //     std::cout << m_max << std::endl;
    //     std::cout << "buff " << m_buffers_map[bufferName] << std::endl;
    // }
  }

  GLuint operator()(const std::string bufferName) { return bufferID(bufferName); }

  GLuint bufferID(const std::string bufferName)
  {
    assert(m_buffers_map.find(bufferName) != m_buffers_map.end());

    return buffers[m_buffers_map[bufferName]];
  }
};

#endif
