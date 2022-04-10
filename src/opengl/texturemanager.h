#pragma once

#ifndef TEXTUREMANAGER_H
#define TEXTUREMANAGER_H

#include <GL/glew.h>

#include <iostream>
#include <map>
#include <vector>

class TextureManager {
private:
    int m_max;
    std::map<std::string, int> m_textures_map;
    std::map<std::string, std::string> m_path_map;

public:
    TextureManager() : m_max(0){};
    ~TextureManager(){};

    std::vector<GLuint> textures;

    int size() const
    {
        return m_max;
    }

    void addTexture(std::string textureName)
    {
        m_textures_map[textureName] = m_max;
        ++m_max;
        textures.resize(m_max);
    }

    void addTexture(std::string textureName, std::string path)
    {
        m_textures_map[textureName] = m_max;
        m_path_map[textureName] = path;
        ++m_max;
        textures.resize(m_max);
    }

    GLuint operator()(const std::string textureName)
    {
        return textureID(textureName);
    }

    GLuint textureID(const std::string textureName)
    {
        assert(m_textures_map.find(textureName) != m_textures_map.end());

        return textures[m_textures_map[textureName]];
    }

    char const *texturePath(const std::string textureName)
    {
        assert(m_path_map.find(textureName) != m_path_map.end());

        return m_path_map[textureName].c_str();
    }
};

#endif
